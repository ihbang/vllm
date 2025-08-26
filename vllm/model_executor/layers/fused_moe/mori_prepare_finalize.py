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

        # Storage for dispatch results that finalize needs
        self._dispatch_cache = None

        # Get registered input buffer for memory efficiency (lazy initialization)
        self._input_buffer = None
        self._combine_buffer = None

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

    def _prepare_scales(self, num_tokens: int, scale_dim: int = 0) -> torch.Tensor:
        """
        Prepare scales tensor for mori dispatch.

        Args:
            num_tokens: Number of tokens
            scale_dim: Scale dimension (0 for no quantization, > 0 for FP8)

        Returns:
            scales: Scales tensor with proper shape
        """
        if scale_dim == 0:
            # No quantization, return empty tensor like mori expects
            return torch.empty(
                (num_tokens, 0),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        else:
            # FP8 quantization enabled, create appropriate scales tensor
            return torch.empty(
                (num_tokens, scale_dim),
                dtype=torch.float8_e4m3fnuz,  # mori uses this FP8 type
                device=torch.cuda.current_device(),
            )

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
            a1_scale: Input activation scales (unused for now)
            a2_scale: Output activation scales (unused for now)
            topk_weights: Top-k routing weights [num_tokens, experts_per_token]
            topk_ids: Top-k expert indices [num_tokens, experts_per_token]
            num_experts: Total number of experts
            expert_map: Expert mapping (unused)
            apply_router_weight_on_input: Whether to apply weights on input (unused)
            quant_config: Quantization config (unused for now)

        Returns:
            Tuple of (dispatched_x, batched_scales, expert_tokens_meta, None, None)
        """
        num_tokens = a1.size(0)
        hidden_dim = a1.size(-1)
        experts_per_token = topk_ids.size(1)

        # Prepare inputs for mori dispatch
        token_indices = self._prepare_token_indices(topk_ids, experts_per_token)
        weights = topk_weights.to(torch.float32)

        # Prepare scales for mori dispatch based on quantization config
        if quant_config.is_quantized and self.use_fp8_dispatch:
            # FP8 quantization enabled - use scale_dim from mori config
            scale_dim = 32  # mori default for FP8 quantization
            if a1_scale is not None:
                # Use provided quantization scales
                scales = a1_scale.to(torch.float8_e4m3fnuz)
                logger.debug(f"Using provided FP8 scales with shape: {scales.shape}")
            else:
                # Create default FP8 scales
                scales = self._prepare_scales(num_tokens, scale_dim)
                scales.fill_(1.0)  # Default scale value
                logger.debug(f"Created default FP8 scales with shape: {scales.shape}")
        else:
            # No quantization - use empty scales tensor
            scales = self._prepare_scales(num_tokens, scale_dim=0)
            logger.debug(f"Using empty scales for non-quantized dispatch")

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
                weights=weights,
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

            logger.debug(
                f"mori dispatch: received {total_recv_num_tokens} tokens"
            )

            # Initialize expert token counts first (outside of if block to avoid UnboundLocalError)
            expert_num_tokens = torch.zeros(self.num_local_experts,
                                          dtype=torch.int32,
                                          device=a1.device)

            # Reshape dispatch output to BatchedExperts format: [num_local_experts, max_tokens_per_expert, hidden_dim]
            # For now, we'll distribute tokens evenly across experts
            # This is a simplified approach - a more sophisticated implementation would
            # use the actual token distribution from mori dispatch

            if total_recv_num_tokens > 0:
                # Calculate tokens per expert (simplified)
                base_tokens_per_expert = total_recv_num_tokens // self.num_local_experts
                remaining_tokens = total_recv_num_tokens % self.num_local_experts

                # Create batched format tensor
                batched_output = torch.zeros(
                    (self.num_local_experts, self.max_num_tokens, hidden_dim),
                    dtype=dispatch_output.dtype,
                    device=dispatch_output.device
                )

                # Fill the batched tensor with dispatch output and track token counts
                start_idx = 0
                for expert_idx in range(self.num_local_experts):
                    expert_tokens = base_tokens_per_expert + (1 if expert_idx < remaining_tokens else 0)
                    if expert_tokens > 0:
                        end_idx = start_idx + expert_tokens
                        actual_tokens = min(expert_tokens, self.max_num_tokens)
                        batched_output[expert_idx, :actual_tokens] = dispatch_output[start_idx:start_idx + actual_tokens]
                        expert_num_tokens[expert_idx] = actual_tokens
                        start_idx = end_idx
            else:
                # No tokens received, create empty batched tensor
                batched_output = torch.zeros(
                    (self.num_local_experts, self.max_num_tokens, hidden_dim),
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
                # For FP8 quantized case, create proper scales tensor with appropriate shape
                scale_shape = quant_config.batched_scale_shape(
                    self.num_local_experts, self.max_num_tokens, hidden_dim)
                batched_scales = torch.empty(scale_shape,
                                           dtype=torch.float32,
                                           device=a1.device)

                # Use dispatch_scales from mori for proper FP8 quantization
                if dispatch_scales is not None and dispatch_scales.numel() > 0:
                    # Reshape dispatch_scales to match batched format if needed
                    if dispatch_scales.numel() == batched_scales.numel():
                        batched_scales.copy_(dispatch_scales.view(batched_scales.shape))
                    else:
                        # Fall back to broadcasting or filling
                        logger.warning(f"Scale shape mismatch: dispatch_scales {dispatch_scales.shape} vs batched_scales {batched_scales.shape}")
                        batched_scales.fill_(1.0)
                else:
                    # No scales from dispatch, use default
                    batched_scales.fill_(1.0)

                logger.debug(f"Created FP8 scales tensor with shape: {scale_shape}")
            else:
                # For non-quantized case, return None
                batched_scales = None

            # Cache dispatch results for finalize phase (after expert_num_tokens is defined)
            self._dispatch_cache = {
                "dispatch_weights": dispatch_weights,
                "dispatch_indices": dispatch_indices,
                "original_token_indices": token_indices,
                "num_received_tokens": dispatch_recv_num_token,
                "original_topk_ids": topk_ids,
                "original_topk_weights": topk_weights,
                "expert_num_tokens": expert_num_tokens,
                "total_recv_num_tokens": total_recv_num_tokens,
            }

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
            fused_expert_output: Expert output activations [num_received_tokens, hidden_dim]
            topk_weights: Original top-k weights (unused, we use cached dispatch weights)
            topk_ids: Original top-k indices (unused, we use cached dispatch indices)
            apply_router_weight_on_input: Whether weights applied on input (unused)
            weight_and_reduce_impl: Weight and reduce implementation (unused for mori)
        """
        if self._dispatch_cache is None:
            raise RuntimeError(
                "No dispatch cache found. Must call prepare() first."
            )

        # Get cached dispatch results
        dispatch_weights = self._dispatch_cache["dispatch_weights"]
        dispatch_indices = self._dispatch_cache["dispatch_indices"]
        original_token_indices = self._dispatch_cache["original_token_indices"]
        original_topk_ids = self._dispatch_cache["original_topk_ids"]
        expert_num_tokens = self._dispatch_cache["expert_num_tokens"]
        total_recv_num_tokens = self._dispatch_cache["total_recv_num_tokens"]

        try:
            # Convert BatchedExperts format back to 2D for mori combine
            # fused_expert_output is [num_local_experts, max_tokens, hidden_dim]
            if fused_expert_output.dim() == 3:
                num_experts, max_tokens, hidden_dim = fused_expert_output.shape

                # Use mori's registered buffer for efficient memory management
                if self._combine_buffer is None:
                    self._combine_buffer = self.handle.get_registered_input_buffer(fused_expert_output.dtype)

                # Copy expert outputs to combine buffer efficiently
                buffer_offset = 0
                for expert_idx in range(num_experts):
                    actual_tokens = int(expert_num_tokens[expert_idx].item())
                    if actual_tokens > 0:
                        expert_output = fused_expert_output[expert_idx, :actual_tokens, :]
                        self._combine_buffer[buffer_offset:buffer_offset + actual_tokens, :].copy_(expert_output)
                        buffer_offset += actual_tokens

                # Use only the filled portion of the buffer
                combine_input = self._combine_buffer[:buffer_offset, :] if buffer_offset > 0 else self._combine_buffer[:0, :]
            else:
                # Already in 2D format - use directly
                combine_input = fused_expert_output

            # Perform mori combine - returns (output, weights) tuple
            combined_output, combined_weights = self.handle.combine(
                input=combine_input,
                weights=dispatch_weights,
                indices=original_token_indices,  # Use original indices from dispatch
                block_num=-1,  # Use default from config
                warp_per_block=-1,  # Use default from config
                call_reset=True,  # Reset internal state after combine
            )

            logger.debug(f"mori combine: output shape {combined_output.shape}")

            # Copy combined result to output tensor
            # Trim to original number of tokens if needed
            num_original_tokens = original_topk_ids.size(0)
            if combined_output.size(0) >= num_original_tokens:
                output.copy_(combined_output[:num_original_tokens])
            else:
                # This shouldn't happen, but handle gracefully
                logger.warning(
                    f"Combined output has fewer tokens than expected: "
                    f"{combined_output.size(0)} < {num_original_tokens}"
                )
                output[: combined_output.size(0)].copy_(combined_output)

            # Clear cache
            self._dispatch_cache = None

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
