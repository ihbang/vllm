# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
mori prepare and finalize module for expert parallelism.
Migration from DeepEP to mori for AMD GPU support.
"""

from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

logger = init_logger(__name__)


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using mori kernels for AMD GPU expert parallelism.

    This class handles the dispatch and combine operations for expert parallelism
    using the mori library, which provides optimized All2All communication
    primitives for AMD GPUs.
    """

    def __init__(
        self,
        handle: Any,  # mori EpDispatchCombineOp from MoriAll2AllManager
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
    ):
        """
        Initialize MoriPrepareAndFinalize.

        Args:
            handle: mori EpDispatchCombineOp instance from All2AllManager
            max_num_tokens: Maximum number of tokens per rank
            num_local_experts: Number of experts on this rank
            num_dispatchers: Number of dispatcher ranks (world size)
            use_fp8_dispatch: Whether to use FP8 quantization during dispatch
        """
        super().__init__()
        assert max_num_tokens > 0
        assert num_local_experts > 0

        self.handle = handle  # mori EpDispatchCombineOp
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers
        self.use_fp8_dispatch = use_fp8_dispatch

        # Minimal storage for finalize (memory optimized)
        self._expert_num_tokens = None  # Only store what we need

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def _prepare_token_indices(
        self, topk_ids: torch.Tensor, experts_per_token: int
    ) -> torch.Tensor:
        """
        Convert topk_ids to mori format token indices.

        Args:
            topk_ids: Top-k expert IDs [num_tokens, experts_per_token]
            experts_per_token: Number of experts per token

        Returns:
            token_indices: Flattened indices for mori [num_tokens * experts_per_token]
        """
        # Convert to local expert indices (mori works with local experts)
        local_expert_ids = topk_ids % self.num_local_experts

        # Flatten for mori format
        token_indices = local_expert_ids.view(-1).to(torch.int32)

        return token_indices

    def _get_mori_config_params(self, input_tensor: torch.Tensor, scales: torch.Tensor) -> dict:
        """
        Get mori configuration parameters based on actual tensor properties.

        Args:
            input_tensor: Input tensor for dispatch
            scales: Scales tensor (can be empty)

        Returns:
            dict: Configuration parameters for mori
        """
        config_params = {
            'data_type': input_tensor.dtype,
            'hidden_dim': input_tensor.size(-1),
            'scale_dim': scales.size(-1) if scales.numel() > 0 else 0,
            'scale_type_size': scales.element_size() if scales.numel() > 0 else 0,
            'max_token_type_size': input_tensor.element_size(),
        }
        
        logger.debug(f"Mori config params: {config_params}")
        return config_params

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[mk.ExpertTokensMetadata],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Prepare inputs for mori dispatch operation.

        Args:
            a1: Input hidden states [num_tokens, hidden_dim]
            a1_scale: Input activation scales
            topk_weights: Top-k routing weights [num_tokens, experts_per_token]
            topk_ids: Top-k expert indices [num_tokens, experts_per_token]
            quant_config: Quantization config

        Returns:
            Tuple of (dispatched_x, batched_scales, expert_tokens_meta, None, None)
        """
        num_tokens = a1.size(0)
        hidden_dim = a1.size(-1)
        experts_per_token = topk_ids.size(1)

        # Prepare inputs for mori dispatch
        token_indices = self._prepare_token_indices(topk_ids, experts_per_token)
        
        # Prepare scales for mori dispatch based on actual quantization state
        if quant_config.is_quantized and self.use_fp8_dispatch and a1_scale is not None:
            # Convert to FP8 type expected by mori (torch.float8_e4m3fnuz)
            scales = a1_scale.to(torch.float8_e4m3fnuz) if a1_scale.dtype != torch.float8_e4m3fnuz else a1_scale
        else:
            # No quantization or no scales provided
            # Empty tensor shape: [num_tokens, 0] to match mori's expectation
            scales = torch.empty(
                (num_tokens, 0),
                dtype=torch.float32,
                device=a1.device,
            )

        try:
            # Perform mori dispatch
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.handle.dispatch(
                input=a1,
                weights=topk_weights,
                scales=scales,
                indices=token_indices,
                block_num=-1,  # Use default from config
                warp_per_block=-1,  # Use default from config
            )

            total_recv_num_tokens = (
                dispatch_recv_num_token[0].item()
                if dispatch_recv_num_token.numel() > 0
                else 0
            )

            # Initialize expert token counts
            expert_num_tokens = torch.zeros(self.num_local_experts,
                                            dtype=torch.int32,  # Required by mori C++ interface
                                            device=a1.device)

            # Reshape dispatch output to BatchedExperts format: [num_local_experts, max_tokens_per_expert, hidden_dim]
            if total_recv_num_tokens > 0:
                # Calculate tokens per expert
                base_tokens_per_expert = total_recv_num_tokens // self.num_local_experts
                remaining_tokens = total_recv_num_tokens % self.num_local_experts

                actual_max_tokens = min(self.max_num_tokens, 
                                      max(base_tokens_per_expert + (1 if remaining_tokens > 0 else 0), 1))
                
                batched_output = torch.zeros(
                    (self.num_local_experts, actual_max_tokens, hidden_dim),
                    dtype=dispatch_output.dtype,
                    device=dispatch_output.device
                )

                # Fill the batched tensor with dispatch output and track token counts
                start_idx = 0
                for expert_idx in range(self.num_local_experts):
                    expert_tokens = base_tokens_per_expert + (1 if expert_idx < remaining_tokens else 0)
                    if expert_tokens > 0:
                        end_idx = start_idx + expert_tokens
                        actual_tokens = min(expert_tokens, actual_max_tokens)
                        if actual_tokens > 0:
                            batched_output[expert_idx, :actual_tokens] = dispatch_output[start_idx:start_idx + actual_tokens]
                        expert_num_tokens[expert_idx] = actual_tokens
                        start_idx = end_idx
            else:
                # No tokens received, create minimal empty tensor
                batched_output = torch.zeros(
                    (self.num_local_experts, 1, hidden_dim),
                    dtype=a1.dtype,
                    device=a1.device
                )

            # Create proper expert tokens metadata
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens,
                expert_num_tokens_cpu=None
            )

            # Create appropriate scales tensor based on quantization config
            if quant_config.is_quantized and self.use_fp8_dispatch:
                # For FP8 quantized case, use dispatch_scales dtype if available
                actual_tokens_dim = batched_output.size(1)
                scale_shape = quant_config.batched_scale_shape(
                    self.num_local_experts, actual_tokens_dim, hidden_dim)
                
                scales_dtype = dispatch_scales.dtype if (dispatch_scales is not None and dispatch_scales.numel() > 0) else torch.float32
                
                batched_scales = torch.empty(scale_shape,
                                             dtype=scales_dtype,
                                             device=a1.device)

                # Use dispatch_scales from mori for proper FP8 quantization
                if dispatch_scales is not None and dispatch_scales.numel() > 0:
                    # Reshape dispatch_scales to match batched format if needed
                    if dispatch_scales.numel() == batched_scales.numel():
                        batched_scales.copy_(dispatch_scales.view(batched_scales.shape))
                    else:
                        # Try to broadcast or fill with appropriate value
                        logger.warning(f"Scale shape mismatch: dispatch_scales {dispatch_scales.shape} vs batched_scales {batched_scales.shape}")
                        # Fill with 1.0 converted to the correct dtype
                        batched_scales.fill_(1.0)
                else:
                    # No scales from dispatch, use default value in correct dtype
                    batched_scales.fill_(1.0)
            else:
                # For non-quantized case, return None
                batched_scales = None

            self._expert_num_tokens = expert_num_tokens

            return (
                batched_output,
                batched_scales,
                expert_tokens_meta,
                None,
                None,
            )

        except Exception as e:
            logger.error(f"mori dispatch failed: {e}")
            raise RuntimeError(f"mori dispatch failed: {e}") from e

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        """
        Finalize expert outputs using mori combine operation.

        Args:
            output: Output tensor to write results [num_original_tokens, hidden_dim]
            fused_expert_output: Expert output activations [varies by format]
            topk_weights: Original top-k weights
            topk_ids: Original top-k indices
        """
        if self._expert_num_tokens is None:
            raise RuntimeError(
                "No expert token data found. Must call prepare() first."
            )

        expert_num_tokens = self._expert_num_tokens
        num_original_tokens = output.size(0)  # Original number of tokens

        try:
            # Convert BatchedExperts format to 2D if needed (following aiter's simpler approach)
            if fused_expert_output.dim() == 3:
                # BatchedExperts format: [num_local_experts, max_tokens, hidden_dim]
                # Convert to 2D by flattening only the used tokens
                num_experts, max_tokens, hidden_dim = fused_expert_output.shape
                
                expert_outputs = []
                for expert_idx in range(num_experts):
                    actual_tokens = int(expert_num_tokens[expert_idx].item())
                    if actual_tokens > 0:
                        expert_output = fused_expert_output[expert_idx, :actual_tokens, :]
                        expert_outputs.append(expert_output)
                
                combine_input = torch.cat(expert_outputs, dim=0) if expert_outputs else torch.empty(
                    (0, hidden_dim), dtype=fused_expert_output.dtype, device=fused_expert_output.device
                )
            else:
                combine_input = fused_expert_output

            combined_output = self.handle.combine(
                input=combine_input,
                weights=topk_weights,
                indices=topk_ids,
                block_num=-1,          # Use default from config
                warp_per_block=-1,     # Use default from config
                call_reset=True,       # Reset internal state after combine
            )

            # Copy result to output tensor, trimmed to original size
            if combined_output.size(0) >= num_original_tokens:
                output.copy_(combined_output[:num_original_tokens])
            else:
                # Handle edge case gracefully
                logger.warning(
                    f"Combined output smaller than expected: "
                    f"{combined_output.size(0)} < {num_original_tokens}"
                )
                output[:combined_output.size(0)].copy_(combined_output)
                output[combined_output.size(0):].zero_()  # Zero remaining

            # Clear cached data
            self._expert_num_tokens = None

        except Exception as e:
            logger.error(f"mori combine failed: {e}")
            raise RuntimeError(f"mori combine failed: {e}") from e

    def __repr__(self) -> str:
        return (
            f"MoriPrepareAndFinalize("
            f"max_tokens={self.max_num_tokens}, "
            f"num_local_experts={self.num_local_experts}, "
            f"num_dispatchers={self.num_dispatchers_})"
        )
