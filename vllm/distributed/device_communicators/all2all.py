# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils import has_deep_ep, has_pplx, has_mori

from .base_device_communicator import All2AllManagerBase, Cache

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
else:
    FusedMoE = None


class NaiveAll2AllManager(All2AllManagerBase):
    """
    A naive implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(self.dp_world_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            self.dp_group.broadcast(buffer[start:end, :], idx)

        return buffer

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu

        hidden_states = self.naive_multicast(hidden_states,
                                             cu_tokens_across_dp_cpu)
        router_logits = self.naive_multicast(router_logits,
                                             cu_tokens_across_dp_cpu)
        return hidden_states, router_logits

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cu_tokens_across_dp_cpu = get_forward_context(
        ).dp_metadata.cu_tokens_across_dp_cpu
        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]

        all_hidden_states = self.dp_group.all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass


class PPLXAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on PPLX kernels.
    """

    def __init__(self, cpu_group):
        assert has_pplx(
        ), "pplx_kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install pplx_kernels."  # noqa
        super().__init__(cpu_group)

        if self.internode:
            # inter-node communication needs nvshmem,
            # intra-node communication uses p2p mapping directly
            from pplx_kernels.nvshmem import (nvshmem_alloc_empty_unique_id,
                                              nvshmem_get_unique_id,
                                              nvshmem_init)
            logger.debug(
                "Initialize NVSHMEM for pplx_kernels: "
                "rank=%d, world size=%d", self.rank, self.world_size)
            uid = nvshmem_get_unique_id(
            ) if self.rank == 0 else nvshmem_alloc_empty_unique_id()
            dist.broadcast(uid,
                           src=dist.get_process_group_ranks(self.cpu_group)[0],
                           group=self.cpu_group)
            logger.debug("PPLX NVSHMEM UID = %s", uid)
            nvshmem_init(uid, self.rank, self.world_size)

        self.handle_cache = Cache()

    def get_handle(self, kwargs):
        import pplx_kernels as pplx
        return self.handle_cache.get_or_create(
            kwargs, pplx.AllToAll.internode
            if self.internode else pplx.AllToAll.intranode)

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()

        if self.internode:
            from pplx_kernels.nvshmem import nvshmem_finalize
            logger.debug("PPLX NVSHMEM finalize")
            nvshmem_finalize()


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group):
        assert has_deep_ep(
        ), "DeepEP kernels not found. Please follow https://github.com/vllm-project/vllm/blob/main/tools/ep_kernels/README.md to install DeepEP kernels."  # noqa
        super().__init__(cpu_group)
        self.handle_cache = Cache()

        # This is the DeepEP default. Stick to it till we can establish
        # reasonable defaults based on profiling.
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        raise NotImplementedError

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        pass


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP High-Throughput kernels.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = 1024 * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode:
            num_rdma_bytes = 1024 * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        return dict(group=self.cpu_group,
                    num_nvl_bytes=num_nvl_bytes,
                    num_rdma_bytes=num_rdma_bytes,
                    low_latency_mode=False,
                    num_qps_per_rank=num_qps_per_rank)

    def get_handle(self, kwargs):

        assert len(kwargs) == 0, (
            "DeepEPHTAll2AllManager expects no arguments. All the required "
            "args are computed in the Manager itself.")

        import deep_ep
        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer)
        # It is dangerous to set num sms outside this function. num_sms is not
        # a part of the hash-key that identifies this object. If we are in a
        # situation where we make objects with different num_sms, the hash key
        # in get_or_create must be updated.
        handle.set_num_sms(self.num_sms)
        return handle


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """
    All2All communication based on DeepEP Low-Latency kernels.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """
        max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import deep_ep

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = 1024 * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts)

        assert num_rdma_bytes is not None
        return dict(group=self.cpu_group,
                    num_nvl_bytes=num_nvl_bytes,
                    num_rdma_bytes=num_rdma_bytes,
                    low_latency_mode=True,
                    num_qps_per_rank=num_qps_per_rank)

    def get_handle(self, kwargs):
        """
        The kwargs for DeepEPLLAll2AllManager is dictated by
        _make_all2all_kwargs.
        """
        import deep_ep
        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer)
        # It is dangerous to set num sms outside this function. num_sms is not
        # a part of the hash-key that identifies this object. If we are in a
        # situation where we make objects with different num_sms, the hash key
        # in get_or_create must be updated.
        handle.set_num_sms(self.num_sms)
        return handle

class MoriAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on mori kernels.
    Migration from DeepEP to mori for AMD GPU support.
    """

    def __init__(self, cpu_group):
        assert has_mori(
        ), "mori not found. Please follow https://github.com/ROCm/mori/blob/main/README.md#installation to install mori."  # noqa

        super().__init__(cpu_group)
        self.handle_cache = Cache()
        self.config = None
        self._op_handles = {}  # Cache for EpDispatchCombineOp instances

        # Initialize mori shmem if not already done
        self._initialize_mori_shmem()

    def _initialize_mori_shmem(self):
        """Initialize mori's shared memory system"""
        import mori.shmem
        import torch.distributed as dist

        try:
            # Register the process group for mori
            world_group = dist.group.WORLD
            if world_group is not None:
                torch._C._distributed_c10d._register_process_group("default", world_group)

            # Initialize mori shared memory
            mori.shmem.shmem_torch_process_group_init("default")
            logger.debug(f"[rank {self.rank}] mori shmem initialized successfully")
        except Exception as e:
            logger.error(f"[rank {self.rank}] mori shmem init failed: {e}")
            raise

    def _make_mori_config(self, max_num_tokens: int, num_local_experts: int,
                          experts_per_token: int, hidden_dim: int,
                          data_type: torch.dtype = torch.bfloat16):
        """Create mori EpDispatchCombineConfig"""
        import mori.ops.dispatch_combine as mori_ops
        from mori.ops.dispatch_combine import EpDispatchCombineKernelType

        # Determine data type size
        dtype_to_size = {
            torch.float32: 4,
            torch.bfloat16: 2,
            torch.float16: 2,
        }
        max_token_type_size = dtype_to_size.get(data_type, 2)

        config = mori_ops.EpDispatchCombineConfig(
            data_type=data_type,
            rank=self.dp_rank,  # Use dp_rank for expert parallelism
            world_size=self.dp_world_size,
            hidden_dim=hidden_dim,
            max_num_inp_token_per_rank=max_num_tokens,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=experts_per_token,

            # Performance tuning parameters (can be optimized later)
            warp_num_per_block=8,  # Good default for MI300X
            block_num=80,          # Good default for MI300X
            max_token_type_size=max_token_type_size,

            # Quantization support (disabled for now)
            scale_dim=0,
            scale_type_size=0,

            # Use internal buffer management
            use_external_inp_buf=False,

            # Determine kernel type based on topology
            kernel_type=(EpDispatchCombineKernelType.InterNode
                        if self.internode
                        else EpDispatchCombineKernelType.IntraNode)
        )

        return config

    def get_handle(self, kwargs):
        """
        Get or create mori operation handle.
        Args:
            kwargs: Dictionary with keys:
                - max_num_tokens: Maximum tokens per DP rank
                - num_local_experts: Number of local experts
                - experts_per_token: Number of experts per token (topk)
                - hidden_dim: Hidden dimension size
                - data_type: Tensor data type (optional, default bfloat16)
        """
        import mori.ops.dispatch_combine as mori_ops

        # Extract parameters
        max_num_tokens = kwargs.get('max_num_tokens')
        num_local_experts = kwargs.get('num_local_experts')
        experts_per_token = kwargs.get('experts_per_token')
        hidden_dim = kwargs.get('hidden_dim')
        data_type = kwargs.get('data_type', torch.bfloat16)

        # Validate required parameters
        if any(param is None for param in [max_num_tokens, num_local_experts,
                                          experts_per_token, hidden_dim]):
            raise ValueError("Missing required parameters for mori handle creation")

        # Create cache key
        cache_key = (max_num_tokens, num_local_experts, experts_per_token,
                    hidden_dim, data_type)

        # Check cache first
        if cache_key in self._op_handles:
            return self._op_handles[cache_key]

        # Create new mori configuration and operation
        config = self._make_mori_config(
            max_num_tokens=max_num_tokens,
            num_local_experts=num_local_experts,
            experts_per_token=experts_per_token,
            hidden_dim=hidden_dim,
            data_type=data_type
        )

        # Create operation handle
        op = mori_ops.EpDispatchCombineOp(config)

        # Cache the handle
        self._op_handles[cache_key] = op

        logger.debug(f"[rank {self.dp_rank}] Created mori handle with config: "
                    f"tokens={max_num_tokens}, experts={num_local_experts}, "
                    f"topk={experts_per_token}, hidden={hidden_dim}")

        return op

    def dispatch(self, hidden_states: torch.Tensor,
                 router_logits: torch.Tensor):
        """
        Dispatch tokens to appropriate experts using mori kernels.

        Args:
            hidden_states: Input token embeddings [num_tokens, hidden_dim]
            router_logits: Router outputs [num_tokens, num_global_experts]

        Returns:
            Tuple of (dispatched_hidden_states, dispatched_router_logits)
        """
        # Get forward context for metadata
        forward_ctx = get_forward_context()
        dp_metadata = forward_ctx.dp_metadata

        # Get handle from cache
        handle = self.get_handle({
            'max_num_tokens': dp_metadata.max_num_tokens_per_dp_rank,
            'num_local_experts': dp_metadata.num_local_experts,
            'experts_per_token': dp_metadata.num_experts_per_token,
            'hidden_dim': hidden_states.size(-1),
            'data_type': hidden_states.dtype
        })

        # Prepare token indices from router logits
        # This converts router logits to expert indices for each token
        token_indices = self._prepare_token_indices(
            router_logits, dp_metadata.num_experts_per_token
        )

        # Prepare weights from router logits
        weights = self._prepare_weights(
            router_logits, dp_metadata.num_experts_per_token
        )

        # Prepare scales (empty for now, no FP8 quantization)
        scales = torch.empty((hidden_states.size(0), 0),
                           dtype=torch.float32,
                           device=hidden_states.device)

        try:
            # Perform mori dispatch
            dispatch_output, dispatch_weights, dispatch_scales, \
            dispatch_indices, dispatch_recv_num_token = handle.dispatch(
                input=hidden_states,
                weights=weights,
                scales=scales,
                indices=token_indices,
                block_num=-1,  # Use default
                warp_per_block=-1  # Use default
            )

            # Store dispatch results in forward context for combine phase
            forward_ctx.mori_dispatch_cache = {
                'dispatch_output': dispatch_output,
                'dispatch_weights': dispatch_weights,
                'dispatch_indices': dispatch_indices,
                'handle': handle,
                'num_received_tokens': dispatch_recv_num_token
            }

            logger.debug(f"[rank {self.dp_rank}] dispatch completed, "
                        f"received {dispatch_recv_num_token[0] if dispatch_recv_num_token.numel() > 0 else 0} tokens")

            return dispatch_output, self._reconstruct_router_logits(
                dispatch_weights, dispatch_indices)

        except Exception as e:
            logger.error(f"[rank {self.dp_rank}] mori dispatch failed: {e}")
            raise

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Combine expert outputs back to original token order using mori kernels.

        Args:
            hidden_states: Expert outputs [num_received_tokens, hidden_dim]

        Returns:
            Combined hidden states [num_local_tokens, hidden_dim]
        """
        # Get cached dispatch results from forward context
        forward_ctx = get_forward_context()
        if not hasattr(forward_ctx, 'mori_dispatch_cache'):
            raise RuntimeError("No mori dispatch cache found. Must call dispatch() first.")

        cache = forward_ctx.mori_dispatch_cache
        handle = cache['handle']
        dispatch_weights = cache['dispatch_weights']
        dispatch_indices = cache['dispatch_indices']

        try:
            # Perform mori combine
            combined_output, combined_weights = handle.combine(
                input=hidden_states,
                weights=dispatch_weights,
                indices=dispatch_indices,
                block_num=-1,  # Use default
                warp_per_block=-1  # Use default
            )

            logger.debug(f"[rank {self.dp_rank}] combine completed")

            # Clean up cache
            delattr(forward_ctx, 'mori_dispatch_cache')

            return combined_output

        except Exception as e:
            logger.error(f"[rank {self.dp_rank}] mori combine failed: {e}")
            raise

    def _prepare_token_indices(self, router_logits: torch.Tensor,
                              experts_per_token: int) -> torch.Tensor:
        """Convert router logits to token indices for mori"""
        num_tokens = router_logits.size(0)
        num_global_experts = router_logits.size(1)

        # Get top-k expert indices
        topk_logits, topk_indices = torch.topk(
            router_logits, experts_per_token, dim=-1
        )

        # Convert global expert indices to local expert indices within DP rank
        num_experts_per_rank = num_global_experts // self.dp_world_size
        local_expert_indices = topk_indices % num_experts_per_rank

        # Flatten for mori format: [num_tokens * experts_per_token]
        token_indices = local_expert_indices.view(-1).to(torch.int32)

        return token_indices

    def _prepare_weights(self, router_logits: torch.Tensor,
                        experts_per_token: int) -> torch.Tensor:
        """Extract top-k weights from router logits"""
        topk_logits, _ = torch.topk(router_logits, experts_per_token, dim=-1)

        # Apply softmax to get routing weights
        routing_weights = torch.softmax(topk_logits, dim=-1)

        return routing_weights.to(torch.float32)

    def _reconstruct_router_logits(self, weights: torch.Tensor,
                                  indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct router logits from dispatched weights and indices"""
        # For now, just return the weights - this may need refinement
        # based on how vllm uses the returned router logits
        return weights

    def destroy(self):
        """Clean up mori resources"""
        try:
            import mori.shmem

            # Clear operation handle cache
            self._op_handles.clear()

            # Finalize mori shared memory
            mori.shmem.shmem_finalize()
            logger.debug(f"[rank {self.dp_rank}] mori resources cleaned up")

        except Exception as e:
            logger.warning(f"[rank {self.dp_rank}] Error during mori cleanup: {e}")
