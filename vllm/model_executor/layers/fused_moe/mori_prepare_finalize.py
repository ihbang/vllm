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
    ):
        """
        Initialize MoriPrepareAndFinalize.

        Args:
            handle: mori EpDispatchCombineOp instance from All2AllManager
            max_num_tokens: Maximum number of tokens per rank
            num_local_experts: Number of experts on this rank
            num_dispatchers: Number of dispatcher ranks (world size)
        """
        super().__init__()
        assert max_num_tokens > 0
        assert num_local_experts > 0

        self.handle = handle  # mori EpDispatchCombineOp
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self.num_dispatchers_ = num_dispatchers

        # Storage for dispatch results that finalize needs
        self._dispatch_cache = None

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

    def _prepare_scales(self, num_tokens: int) -> torch.Tensor:
        """
        Prepare empty scales tensor for mori (no quantization for now).

        Args:
            num_tokens: Number of tokens

        Returns:
            scales: Empty scales tensor
        """
        return torch.empty(
            (num_tokens, 0),
            dtype=torch.float32,
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
        extra_prepare_args: Optional[dict[str, Any]],
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
            extra_prepare_args: Extra arguments (unused)

        Returns:
            Tuple of (dispatched_x, batched_scales, expert_tokens_meta, None, None)
        """
        num_tokens = a1.size(0)
        hidden_dim = a1.size(-1)
        experts_per_token = topk_ids.size(1)

        # Prepare inputs for mori dispatch
        token_indices = self._prepare_token_indices(topk_ids, experts_per_token)
        weights = topk_weights.to(torch.float32)
        scales = self._prepare_scales(num_tokens)

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

            # Cache dispatch results for finalize phase
            self._dispatch_cache = {
                "dispatch_weights": dispatch_weights,
                "dispatch_indices": dispatch_indices,
                "original_token_indices": token_indices,
                "num_received_tokens": dispatch_recv_num_token,
                "original_topk_ids": topk_ids,
                "original_topk_weights": topk_weights,
            }

            total_recv_num_tokens = (
                dispatch_recv_num_token[0].item()
                if dispatch_recv_num_token.numel() > 0
                else 0
            )

            logger.debug(
                f"mori dispatch: received {total_recv_num_tokens} tokens"
            )

            # Return trimmed output and metadata
            expert_tokens_meta = None  # mori doesn't use this metadata format
            batched_scales = None  # No quantization for now

            return (
                dispatch_output[:total_recv_num_tokens],
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
        extra_finalize_args: Optional[dict[str, Any]],
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
            extra_finalize_args: Extra arguments (unused)
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

        try:
            # Perform mori combine
            combined_output, combined_weights = self.handle.combine(
                input=fused_expert_output,
                weights=dispatch_weights,
                indices=original_token_indices,  # Use original indices from dispatch
                block_num=-1,  # Use default from config
                warp_per_block=-1,  # Use default from config
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
