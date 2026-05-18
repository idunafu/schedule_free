# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim
from ._quantized_state import (
    BNB_DYNAMIC_BACKEND,
    dequantize_exp_avg_sq,
    init_exp_avg_sq_state,
    quantize_exp_avg_sq,
    validate_block_size,
    validate_quant_backend,
)
try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT : TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class RAdamScheduleFree8bit(torch.optim.Optimizer):
    r"""
    Schedule-Free RAdam with an 8-bit second moment state.

    This optimizer stores the RAdam second moment estimate (`exp_avg_sq`) in a
    block-wise uint8 representation. The Schedule-Free `z` sequence remains in
    the parameter dtype because quantizing it changes the optimized iterate
    directly.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        r (float): Use polynomial weighting in the average
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Present for API compatibility. foreach=True is not
            supported because the 8-bit state is dequantized per tensor.
        silent_sgd_phase (bool): If True, the optimizer will not use the first SGD phase of RAdam.
            This means that the optimizer will not update model parameters during the early training
            steps (e.g., < 5 when β_2 = 0.999), but just update the momentum values of the optimizer.
            This helps stabilize training by ensuring smoother warmup behavior and more reliable
            calculation of the moving average coefficient (`ckp1`). Recommended to set to True
            (default True).
        block_size (int): Number of values that share one quantization scale
            (default 4096).
        min_8bit_size (int): Parameters with fewer values keep exp_avg_sq in
            full precision to avoid overhead on tiny tensors (default 4096).
        quant_backend (str): 8-bit state backend. "bnb_dynamic" uses a
            bitsandbytes-style unsigned dynamic qmap with per-block absmax.
            "torch_linear" keeps the previous linear uint8 scaling backend.
            (default "bnb_dynamic").
    """

    def __init__(self,
                 params: ParamsT,
                 lr: Union[float, torch.Tensor] = 0.0025,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 r: float = 0.0,
                 weight_lr_power: float = 2.0,
                 foreach: Optional[bool] = False,
                 silent_sgd_phase: bool = True,
                 block_size: int = 4096,
                 min_8bit_size: int = 4096,
                 quant_backend: str = BNB_DYNAMIC_BACKEND
                 ):

        if foreach:
            raise ValueError("RAdamScheduleFree8bit does not support foreach=True")
        validate_quant_backend(quant_backend)
        validate_block_size(block_size, quant_backend)
        if min_8bit_size < 0:
            raise ValueError("min_8bit_size must be non-negative")

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        r=r,
                        k=0,
                        train_mode=False,
                        weight_sum=0.0,
                        lr_max=-1.0,
                        scheduled_lr=0.0,
                        weight_lr_power=weight_lr_power,
                        weight_decay=weight_decay,
                        foreach=False,
                        silent_sgd_phase=silent_sgd_phase,
                        block_size=block_size,
                        min_8bit_size=min_8bit_size,
                        quant_backend=quant_backend)
        super().__init__(params, defaults)

    @classmethod
    def _init_exp_avg_sq(cls, state: Dict[str, Any], p: torch.Tensor,
                         block_size: int, min_8bit_size: int,
                         quant_backend: str) -> None:
        init_exp_avg_sq_state(state, p, block_size, min_8bit_size, quant_backend)

    @staticmethod
    def _dequantize_exp_avg_sq(state: Dict[str, Any], p: torch.Tensor) -> torch.Tensor:
        return dequantize_exp_avg_sq(state, p)

    @staticmethod
    def _quantize_exp_avg_sq(state: Dict[str, Any], exp_avg_sq: torch.Tensor) -> None:
        quantize_exp_avg_sq(state, exp_avg_sq)

    def load_state_dict(self, state_dict):  # type: ignore[override]
        result = super().load_state_dict(state_dict)
        for group in self.param_groups:
            group.setdefault("quant_backend", BNB_DYNAMIC_BACKEND)
        for state in self.state.values():
            if ("exp_avg_sq_q" in state and
                    state.get("exp_avg_sq_quant_backend") not in ("bnb_dynamic", "torch_linear")):
                if "exp_avg_sq_absmax" in state:
                    state["exp_avg_sq_quant_backend"] = "bnb_dynamic"
                else:
                    state["exp_avg_sq_quant_backend"] = "torch_linear"
            if "exp_avg_sq_q" in state and state["exp_avg_sq_q"].dtype != torch.uint8:
                state["exp_avg_sq_q"] = state["exp_avg_sq_q"].to(dtype=torch.uint8)
            if "exp_avg_sq_absmax" in state:
                state["exp_avg_sq_absmax"] = state["exp_avg_sq_absmax"].to(dtype=torch.float32)
            if "exp_avg_sq_scale" in state:
                state["exp_avg_sq_scale"] = state["exp_avg_sq_scale"].to(dtype=torch.float32)
        return result

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to x
                        p.lerp_(end=state["z"].to(p.device), weight=1 - 1 / beta1)
                group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p to y
                        p.lerp_(end=state["z"].to(p.device), weight=1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.param_groups[0]["train_mode"]:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the "
                "optimizer. See documentation for details."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            silent_sgd_phase = group["silent_sgd_phase"]
            k = group["k"]  # current steps
            step = k + 1
            r = group["r"]
            weight_lr_power = group["weight_lr_power"]
            block_size = group["block_size"]
            min_8bit_size = group["min_8bit_size"]
            quant_backend = group["quant_backend"]

            beta2_t = beta2**step
            bias_correction2 = 1 - beta2_t

            # maximum length of the approximated SMA
            rho_inf = 2 / (1 - beta2) - 1
            # compute the length of the approximated SMA
            rho_t = rho_inf - 2 * step * beta2_t / bias_correction2
            rect = (
                ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                if rho_t > 4.0
                else float(not silent_sgd_phase)
            )

            lr = group["lr"] * rect
            group["scheduled_lr"] = lr  # For logging purposes

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = (step**r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            adaptive_y_lr = lr * (beta1 * (1 - ckp1) - 1)
            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                state = self.state[p]
                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    self._init_exp_avg_sq(state, p, block_size, min_8bit_size, quant_backend)

            for p in active_p:
                y = p  # Notation to match theory
                grad = p.grad

                state = self.state[p]

                z = state["z"]
                if "exp_avg_sq_q" in state:
                    exp_avg_sq = self._dequantize_exp_avg_sq(state, p)
                    quantized_state = True
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    quantized_state = False

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if quantized_state:
                    self._quantize_exp_avg_sq(state, exp_avg_sq)

                if rho_t > 4.0:
                    # Adam step
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    # Reuse grad buffer for memory efficiency
                    grad_normalized = grad.div_(denom)
                else:
                    # Fall back to SGD (or nothing)
                    grad_normalized = grad

                # Weight decay calculated at y
                if decay != 0:
                    grad_normalized.add_(y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                y.lerp_(end=z, weight=ckp1)
                y.add_(grad_normalized, alpha=adaptive_y_lr)

                # z step
                z.sub_(grad_normalized, alpha=lr)

            group["k"] = k + 1
        return loss
