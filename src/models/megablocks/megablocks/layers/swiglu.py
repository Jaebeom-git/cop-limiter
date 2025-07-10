import torch
import torch.nn as nn
import stk
from functools import partial

from megablocks.layers import common, mpu
from megablocks.layers.arguments import Arguments
from megablocks.layers.mlp import (
    SparseMLP,
    SharedMLP,
    create_dmoe_expert_weights,
    resolve_dtensor,
    scale_gradient,
)
from megablocks.layers.activation_fn import act_fn


def swish_with_beta(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Swish activation with a learnable beta parameter."""
    return x * torch.sigmoid(beta * x)


class SparseSwiGLU(SparseMLP):
    """Sparse version of SwiGLU (Swish + GLU) using `act_fn`.
    
    1) x1 = sdd(x, w1^T)
    2) x2 = sdd(x, v1^T)
    3) x1 = Swish(x1)  (learnable beta)
    4) x1 = x1 * x2    (GLU gate)
    5) out = dsd(x1, w2)
    """

    def __init__(self, args: Arguments):
        super().__init__(args)  # SparseMLP.__init__에서 w1, w2가 생성됨

        # GLU에서 gate 역할을 하는 v1 파라미터 추가
        self.v1 = nn.Parameter(
            torch.empty(
                self._num_rows_per_rank,
                args.hidden_size,
                device=args.device,
                dtype=common.dtype(args),
            )
        )
        with torch.no_grad():
            self.v1.copy_(
                create_dmoe_expert_weights(
                    args,
                    args.moe_num_experts,
                    args.ffn_hidden_size,
                    args.hidden_size,
                    args.init_method,
                ),
            )

        # Swish의 learnable beta 파라미터
        self.beta = nn.Parameter(
            torch.tensor(1.0, dtype=common.dtype(args), device=args.device)
        )

        # expert model parallel attributes
        self._should_set_parallelism_attribute = args.moe_expert_model_parallelism
        mpu.set_expert_model_parallel_attributes(
            self.v1,
            self._should_set_parallelism_attribute,
        )

    def forward(self, x: stk.Matrix, topo) -> stk.Matrix:
        """
        Args:
            x:    STK Matrix (batch_size, hidden_size) 형태의 sparse 텐서
            topo: sparse 연산(sdd/dsd)에 필요한 토폴로지 정보
        Returns:
            out:  STK Matrix
        """
        # 1) Gradient scaling (MoE expert model 병렬 시 적용)
        w1 = scale_gradient(self.w1, self.gradient_scale) if self.gradient_scale else self.w1
        v1 = scale_gradient(self.v1, self.gradient_scale) if self.gradient_scale else self.v1
        w2 = scale_gradient(self.w2, self.gradient_scale) if self.gradient_scale else self.w2

        # DTensor(if any) -> local tensor
        w1, v1, w2 = resolve_dtensor(w1), resolve_dtensor(v1), resolve_dtensor(w2)

        # 2) Sparse MatMul: x1, x2
        #    x1 = x * w1^T,  x2 = x * v1^T
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)

        # 3) Swish 적용 (act_fn 사용)
        #    swish_with_beta(x, beta) = x * sigmoid(beta * x)
        swish_fn = partial(swish_with_beta, beta=self.beta)
        x1 = act_fn(x1, swish_fn)  # x1이 STK Matrix로 반환

        # 4) GLU: x1 * x2
        #    (Swish된 x1) * (원본 x2)
        x1 = stk.ops.mul(x1, x2)

        # 5) 최종 프로젝션: dsd(x1, w2)
        out = stk.ops.dsd(x1, w2)
        return out


class SharedSwiGLU(SharedMLP):
    """Shared Expert MLP with SwiGLU:
       up_proj(x) -> Swish(learnable beta) -> gate_proj(x) -> GLU -> down_proj
    """

    def __init__(self, args: Arguments):
        super().__init__(args)
        self.gate_proj = args.fc_cls(
            args.hidden_size,
            self.args.shared_expert_hidden_size,
            **self.fc_kwargs,
        )
        self.beta = nn.Parameter(
            torch.tensor(1.0, device=args.device, dtype=self.up_proj.weight.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) up_proj(x)
        up = self.up_proj(x)

        # 2) Swish (x -> x * sigmoid(beta * x))
        swish_up = swish_with_beta(up, self.beta)

        # 3) GLU: swish_up * gate_proj(x)
        gate = self.gate_proj(x)
        glu_out = swish_up * gate

        # 4) down_proj
        return self.down_proj(glu_out)