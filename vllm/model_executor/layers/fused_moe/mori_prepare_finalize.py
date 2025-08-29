# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
mori prepare and finalize module for expert parallelism.
Migration from DeepEP to mori for AMD GPU support.
"""

from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate)
from vllm.logger import init_logger

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

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

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
        Optimized to minimize host-device synchronization points.

        Args:
            a1: Input hidden states [num_tokens, hidden_dim]
            a1_scale: Input activation scales
            topk_weights: Top-k routing weights [num_experts, experts_per_token]
            topk_ids: Top-k expert indices [num_experts, experts_per_token]
            quant_config: Quantization config

        Returns:
            Tuple of (dispatched_x, batched_scales, expert_tokens_meta, dispatch_indices, dispatch_weights)
            where dispatched_x is in Standard format (2D tensor)
        """
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
                scales=None,
                indices=topk_ids,
            )

            if dispatch_recv_num_token.numel() > 0:
                max_recv_tokens = dispatch_recv_num_token.max()

                actual_max = min(max_recv_tokens.item(), dispatch_output.size(0))
                standard_output = dispatch_output[:actual_max]
                valid_dispatch_weights = dispatch_weights[:actual_max] 
                valid_dispatch_indices = dispatch_indices[:actual_max]
            else:
                # No tokens received
                standard_output = torch.empty((0, a1.size(1)), dtype=a1.dtype, device=a1.device)
                valid_dispatch_weights = torch.empty((0, topk_weights.size(1)), dtype=topk_weights.dtype, device=a1.device) 
                valid_dispatch_indices = torch.empty((0, topk_ids.size(1)), dtype=topk_ids.dtype, device=a1.device)
                actual_max = 0

            if not hasattr(self, '_expert_tokens_buffer'):
                self._expert_tokens_buffer = torch.zeros(self.num_local_experts,
                                                       dtype=torch.int32,
                                                       device=a1.device)
            else:
                self._expert_tokens_buffer.zero_()
            
            expert_num_tokens = self._expert_tokens_buffer

            if actual_max > 0:
                # Estimate expert token distribution using GPU operations where possible
                base_tokens_per_expert = actual_max // self.num_local_experts
                remaining_tokens = actual_max % self.num_local_experts
                
                if base_tokens_per_expert > 0:
                    expert_num_tokens.fill_(base_tokens_per_expert)
                
                if remaining_tokens > 0:
                    expert_num_tokens[:remaining_tokens] += 1

            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens,
                expert_num_tokens_cpu=None
            )

            # Store for finalize() - avoid recomputation
            self._last_dispatch_size = actual_max

            return (
                standard_output,
                None,
                expert_tokens_meta,
                valid_dispatch_indices,
                valid_dispatch_weights,
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
            fused_expert_output: Expert output activations in Standard format (2D tensor)
            topk_weights: Original top-k weights
            topk_ids: Original top-k indices
        """
        assert self.handle is not None

        num_original_tokens = output.size(0)  # Original number of tokens

        try:
            # fused_expert_output can have 0 tokens - This happens when none of the
            # tokens from the all2all reach this EP rank.
            if fused_expert_output.numel() != 0:
                if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                    weight_and_reduce_impl = TopKWeightAndReduceContiguous()
                fused_expert_output = weight_and_reduce_impl.apply(
                    output=None,
                    fused_expert_output=fused_expert_output,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                )

            combined_output, combined_weights = self.handle.combine(
                input=fused_expert_output,
                weights=topk_weights,
                indices=topk_ids,
            )

            output.copy_(combined_output[:num_original_tokens], non_blocking=True)

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
