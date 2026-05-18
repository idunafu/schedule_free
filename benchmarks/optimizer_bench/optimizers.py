from __future__ import annotations

import importlib
from typing import Dict, Iterable

import torch

import schedulefree


class OptionalOptimizerUnavailable(RuntimeError):
    pass


class CompatibleScheduleFreeWrapper:
    def __init__(
        self,
        base: torch.optim.Optimizer,
        weight_decay_at_y: float = 0.0,
        momentum: float = 0.9,
        weight_lr_power: float = 2.0,
        r: float = 0.0,
    ) -> None:
        self.base = base
        self.weight_decay_at_y = weight_decay_at_y
        self.weight_lr_power = weight_lr_power
        self.r = r
        self.momentum = momentum
        self.train_mode = False
        self.schedulefree_state: Dict[torch.Tensor, Dict[str, torch.Tensor]] = {}

    def add_param_group(self, param_group):
        return self.base.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.schedulefree_state = state_dict.get("schedulefree_state", {})
        return self.base.load_state_dict(state_dict["base"])

    def state_dict(self):
        return {
            "base": self.base.state_dict(),
            "schedulefree_state": self.schedulefree_state,
        }

    def zero_grad(self, set_to_none: bool = True):
        return self.base.zero_grad(set_to_none)

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def state(self):
        return {
            "base": self.base.state,
            "schedulefree": self.schedulefree_state,
        }

    @torch.no_grad()
    def eval(self) -> None:
        if self.train_mode:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.schedulefree_state.get(p)
                    if state is not None and "z" in state:
                        # Set p to x.
                        p.lerp_(end=state["z"], weight=1 - 1 / self.momentum)
        self.train_mode = False

    @torch.no_grad()
    def train(self) -> None:
        if not self.train_mode:
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.schedulefree_state.get(p)
                    if state is not None and "z" in state:
                        # Set p to y.
                        p.lerp_(end=state["z"], weight=1 - self.momentum)
        self.train_mode = True

    @staticmethod
    def swap(x: torch.Tensor, y: torch.Tensor) -> None:
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the optimizer."
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.schedulefree_state.setdefault(p, {})

                if "z" not in state:
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)

                z = state["z"]

                if self.weight_decay_at_y != 0.0:
                    z.sub_(p, alpha=lr * self.weight_decay_at_y)
                    p.sub_(p, alpha=lr * self.weight_decay_at_y * (1 - self.momentum))

                # Convert y -> x, then swap so the base optimizer steps z.
                p.lerp_(end=z, weight=1 - 1 / self.momentum)
                self.swap(z, p)

        self.base.step()

        for group in self.param_groups:
            k = group.get("k", 0)
            d = group.get("d", 1.0)
            lr = group["lr"] * d
            lr_max = group["lr_max"] = max(lr, group.get("lr_max", 0))
            weight = ((k + 1) ** self.r) * (lr_max ** self.weight_lr_power)
            weight_sum = group["weight_sum"] = group.get("weight_sum", 0.0) + weight
            ckp1 = weight / weight_sum

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.schedulefree_state[p]
                z = state["z"]

                # Restore x from the wrapper state, update x, then set p to y.
                self.swap(z, p)
                p.lerp_(end=z, weight=ckp1)
                p.lerp_(end=z, weight=1 - self.momentum)

            group["k"] = k + 1

        return loss


def make_optimizer(name: str, params: Iterable[torch.Tensor], args, device: torch.device):
    betas = (args.beta1, args.beta2)
    if name == "adamw":
        return schedulefree.AdamWScheduleFree(
            params,
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            foreach=False,
        )
    if name == "adamw8bit":
        return schedulefree.AdamWScheduleFree8bit(
            params,
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            block_size=args.block_size,
            min_8bit_size=args.min_8bit_size,
            quant_backend=args.quant_backend,
        )
    if name == "radam":
        return schedulefree.RAdamScheduleFree(
            params,
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
            foreach=False,
        )
    if name == "radam8bit":
        return schedulefree.RAdamScheduleFree8bit(
            params,
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
            block_size=args.block_size,
            min_8bit_size=args.min_8bit_size,
            quant_backend=args.quant_backend,
        )
    if name == "wrapped-bnb-adamw8bit":
        if device.type != "cuda":
            raise OptionalOptimizerUnavailable("bitsandbytes AdamW8bit comparison requires a CUDA device.")
        try:
            bnb = importlib.import_module("bitsandbytes")
        except ImportError as exc:
            raise OptionalOptimizerUnavailable(
                "bitsandbytes is not installed; skipping wrapped-bnb-adamw8bit."
            ) from exc
        base = bnb.optim.AdamW8bit(
            params,
            lr=args.lr,
            betas=(args.wrapper_beta1, args.beta2),
            weight_decay=0.0,
        )
        return CompatibleScheduleFreeWrapper(
            base,
            momentum=args.beta1,
            weight_decay_at_y=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")
