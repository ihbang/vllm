"""
Aiter-based expert processing for Mori integration.
"""

from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_fused_experts,
)


class AiterExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Aiter-based expert processing that works with Mori dispatch/combine.

    This class bridges Mori's all2all communication with Aiter's optimized
    expert computation kernels for AMD GPUs.
    """

    def __init__(
        self,
        max_num_tokens: int,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        use_mxfp4_w4a4: bool = False,
        per_act_token_quant: bool = False,
        block_shape: Optional[list[int]] = None,
    ):
        super().__init__(
            FusedMoEQuantConfig.make(
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                use_mxfp4_w4a4=use_mxfp4_w4a4,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
            )
        )

        self.use_fp8_w8a8 = use_fp8_w8a8
        self.use_int8_w8a8 = use_int8_w8a8
        self.use_int4_w4a16 = use_int4_w4a16
        self.use_int8_w8a16 = use_int8_w8a16
        self.use_mxfp4_w4a4 = use_mxfp4_w4a4
        self.max_num_tokens = max_num_tokens
        self._per_act_token_quant = per_act_token_quant

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        """Aiter expects Standard format for both input and output."""
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        """Aiter kernels support chunking."""
        return True

    def supports_expert_map(self) -> bool:
        """Aiter kernels support expert mapping."""
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        """Aiter handles weight and reduce internally."""
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        """
        Aiter kernels manage memory internally, so minimal workspace is needed.
        """
        # Return minimal shapes since Aiter handles memory internally
        workspace2 = ()  # No intermediate workspace needed
        output_shape = aq.shape
        workspace13 = output_shape
        workspace_dtype = a.dtype
        return (workspace13, workspace2, output_shape, workspace_dtype)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
        apply_router_weight_on_input: bool,
    ):
        """
        Process expert computation using Aiter kernels.
        Works with pre-dispatched tokens from Mori all2all.
        """
        # Call Aiter fused MoE expert processing
        result = rocm_aiter_fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=self.use_fp8_w8a8,
            per_channel_quant=self._per_act_token_quant,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1q_scale,  # Map a1q_scale -> a1_scale for Aiter
            a2_scale=a2_scale,
            block_shape=self.block_shape,
            expert_map=expert_map,
            expert_num_tokens=expert_tokens_meta.expert_num_tokens,
            output_dtype=output.dtype,
        )

        # Copy result to output tensor
        output.copy_(result)
