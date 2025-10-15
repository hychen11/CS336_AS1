from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # Increment iteration number.
        return loss


# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1e3)

# for t in range(10):
#     opt.zero_grad()  # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean()  # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward()  # Run backward pass, which computes gradients.
#     opt.step()  # Run optimizer step.


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            # here have hyperparameters (beta1,beta2) pairs!
            betas1, betas2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead")
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # m is first moment vector
                # v is second moment vector
                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]
                grad = p.grad.data

                m.mul_(betas1).add_(grad, alpha=1 - betas1)
                v.mul_(betas2).addcmul_(grad, grad, value=1 - betas2)

                bias_correction1 = 1 - betas1 ** step
                bias_correction2 = 1 - betas2 ** step
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                # Apply weight decay first
                if weight_decay != 0:
                    p.data.mul_(1-lr*weight_decay)
                    
                # Update the parameters second
                lr_t = lr * bias_correction2_sqrt / bias_correction1
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

        return loss
