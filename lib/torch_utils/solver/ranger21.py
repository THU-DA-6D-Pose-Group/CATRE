# Modified from: https://github.com/lessw2020/Ranger21/blob/main/ranger21/ranger21.py
# removed the lr schedule related code

# Ranger21 - @lessw2020  and @NestorDemeure
# with contributions from:
# @BrianPugh
# @Kayuksel
# @TheZothen

# core components based on:

# MADGRAD: https://arxiv.org/abs/2101.11075

# warmup:  https://arxiv.org/abs/1910.04209v3

# stable weight decay: https://arxiv.org/abs/2011.11152v3

# Gradient Centralization: https://arxiv.org/abs/2004.01461v2

# positive negative momentum:  https://arxiv.org/abs/2103.17182

# adaptive gradient clipping: https://arxiv.org/abs/2102.06171)
#   big thanks to @kayuksel for suggestion to include agc, and initial code,
#   @lucidrains and @rwightman for additional code impl reference

# softplus transformation to denom:  https://arxiv.org/abs/1908.00700

# lookahead:
# Lookahead Optimizer: https://arxiv.org/abs/1907.08610

# norm loss: https://arxiv.org/abs/2103.06583v1
# big thanks to Theodoros Georgiou for TF code implementation, and he and team for inventing norm loss

# flat lr + cosine decay: original work 2019 (fastai team)

# Chebyshev fractal steps:

# This space for rent - send in your improvements!


import torch
import torch.optim as TO
import torch.nn.functional as F

import math
import collections

import copy
from torch import linalg as LA

import numpy as np


def normalize_gradient(x, use_channels=False, epsilon=1e-8):
    """use stdev to normalize gradients."""
    size = x.dim()
    # print(f"size = {size}")

    if (size > 1) and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True) + epsilon
        # print(f"s = {s}")
        x.div_(s)  # , keepdim=True)

    elif torch.numel(x) > 2:
        s = x.std() + epsilon
        x.div_(s)  # , keepdim=True)
    return x


def centralize_gradient(x, gc_conv_only=False):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization"""

    size = x.dim()
    # print(f"size = {size}")

    if gc_conv_only:
        if size > 3:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    else:
        if size > 1:
            x.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))
    return x


class Ranger21(TO.Optimizer):
    def __init__(
        self,
        params,
        lr,
        lookahead_active=True,
        lookahead_mergetime=5,
        lookahead_blending_alpha=0.5,
        lookahead_load_at_validation=False,
        use_madgrad=False,
        use_adabelief=False,
        softplus=True,
        beta_softplus=50,
        using_gc=True,
        using_normgc=True,
        gc_conv_only=False,
        normloss_active=True,
        normloss_factor=1e-4,
        use_adaptive_gradient_clipping=True,
        agc_clipping_value=1e-2,
        agc_eps=1e-3,
        betas=(0.9, 0.999),  # temp for checking tuned warmups
        momentum_type="pnm",
        pnm_momentum_factor=1.0,
        momentum=0.9,
        eps=1e-8,
        weight_decay=1e-4,
        decay_type="stable",
        logging_active=True,
    ):

        # todo - checks on incoming params
        defaults = dict(lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # core
        self.logging = logging_active

        # engine
        self.use_madgrad = use_madgrad

        if not self.use_madgrad:
            self.core_engine = "AdamW"
        else:
            self.core_engine = "madgrad"

        # ada belief:
        self.use_adabelief = use_adabelief

        # eps
        self.eps = eps

        # softplus for denom
        self.softplus = softplus
        self.beta_softplus = beta_softplus

        # norm loss
        self.normloss_active = normloss_active
        self.normloss_factor = normloss_factor

        # lookahead
        self.lookahead_active = lookahead_active
        self.lookahead_mergetime = lookahead_mergetime
        self.lookahead_step = 0
        self.lookahead_alpha = lookahead_blending_alpha

        self.lookahead_validation_load = lookahead_load_at_validation

        # agc
        self.agc_active = use_adaptive_gradient_clipping
        self.agc_clip_val = agc_clipping_value
        self.agc_eps = agc_eps

        # lr
        self.starting_lr = lr

        self.use_gc = using_gc
        self.use_gcnorm = using_normgc
        self.gc_conv_only = gc_conv_only

        # epochs
        self.epoch_count = 0

        # momentum
        self.momentum_pnm = momentum_type == "pnm"

        self.pnm_momentum = pnm_momentum_factor

        # decay
        self.decay = weight_decay
        self.decay_type = decay_type
        self.param_size = 0

        # logging - need to update things before moving these 2 into self.logging toggle

        if self.logging:
            self.tracking_variance_sum = []
            self.tracking_variance_normalized = []

        # print out initial settings to make usage easier

        self.show_settings()

    def __setstate__(self, state):
        super().__setstate__(state)

    def show_settings(self):
        print(f"Ranger21 optimizer ready with following settings:\n")
        print(f"Core optimizer = {self.core_engine}")
        print(f"Learning rate of {self.starting_lr}\n")

        if self.use_adabelief:
            print(f"using AdaBelief for variance computation")

        if self.lookahead_active:
            print(
                f"Lookahead active, merging every {self.lookahead_mergetime} steps, with blend factor of {self.lookahead_alpha}"
            )
        if self.normloss_active:
            print(f"Norm Loss active, factor = {self.normloss_factor}")
        if self.decay:
            print(f"Stable weight decay of {self.decay}")

        if self.use_gc:
            print(f"Gradient Centralization = On\n")
        else:
            print("Gradient Centralization = Off\n")

        print(f"Adaptive Gradient Clipping = {self.agc_active}")
        if self.agc_active:
            print(f"\tclipping value of {self.agc_clip_val}")
            print(f"\tsteps for clipping = {self.agc_eps}")

    # lookahead functions
    def clear_cache(self):
        """clears the lookahead cached params."""

        print(f"clearing lookahead cache...")
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                try:
                    la_params = param_state["lookahead_params"]
                except:
                    print(f"no lookahead cache present.")
                    return

                if len(la_params):
                    param_state["lookahead_params"] = torch.zeros_like(p.data)
        print(f"lookahead cache cleared")

    def clear_and_load_backup(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]

    def backup_and_load_cache(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["lookahead_params"])

    def unit_norm(self, x):
        """axis-based Euclidean norm."""
        # verify shape
        keepdim = True
        dim = None

        xlen = len(x.shape)
        # print(f"xlen = {xlen}")

        if xlen <= 1:
            keepdim = False
        elif xlen in (2, 3):  # linear layers
            dim = 1
        elif xlen == 4:  # conv kernels
            dim = (1, 2, 3)
        else:
            dim = tuple([x for x in range(1, xlen)])  # create 1,..., xlen-1 tuple, while avoiding last dim ...

        return x.norm(dim=dim, keepdim=keepdim, p=2.0)

    def agc(self, p):
        """clip gradient values in excess of the unitwise norm.

        the hardcoded 1e-6 is simple stop from div by zero and no
        relation to standard optimizer eps
        """

        # params = [p for p in parameters if p.grad is not None]
        # if not params:
        #    return

        # for p in params:
        p_norm = self.unit_norm(p).clamp_(self.agc_eps)
        g_norm = self.unit_norm(p.grad)

        max_norm = p_norm * self.agc_clip_val

        clipped_grad = p.grad * (max_norm / g_norm.clamp(min=1e-6))

        new_grads = torch.where(g_norm > max_norm, clipped_grad, p.grad)
        p.grad.detach().copy_(new_grads)

    def get_variance(self):
        return self.tracking_variance_sum

    def get_state_values(self, group, state):
        beta1, beta2 = group["betas"]
        mean_avg = state["mean_avg"]
        variance_avg = state["variance_avg"]

        return beta1, beta2, mean_avg, variance_avg

    # @staticmethod
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None and isinstance(closure, collections.Callable):
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        variance_ma_sum = 0.0

        # phase 1 - accumulate all of the variance_ma_sum to use in stable weight decay

        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # if not self.param_size:
                param_size += p.numel()

                # apply agc if enabled
                if self.agc_active:
                    self.agc(p)

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("sparse matrix not supported atm")

                state = self.state[p]
                momentum = group["momentum"]

                # State initialization
                if len(state) == 0:
                    # print("init state")
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["variance_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if self.lookahead_active:
                        state["lookahead_params"] = torch.zeros_like(p.data)
                        state["lookahead_params"].copy_(p.data)

                    if self.use_adabelief:
                        state["variance_ma_belief"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.momentum_pnm:
                        state["neg_grad_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_variance_ma"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # Cumulative products of beta1
                    # state["beta1_prod"] = torch.ones_like(
                    #    p.data, memory_format=torch.preserve_format
                    # )

                # centralize gradients
                if self.use_gc:
                    grad = centralize_gradient(
                        grad,
                        gc_conv_only=self.gc_conv_only,
                    )
                if self.use_gcnorm:
                    grad = normalize_gradient(grad)
                # else:
                #    grad = uncentralized_grad

                # phase 1, variance computations

                state["step"] += 1

                step = state["step"]
                lr = group["lr"]

                beta1, beta2 = group["betas"]
                grad_ma = state["grad_ma"]

                bias_correction2 = 1 - beta2 ** state["step"]
                # print(f"bias2 = {bias_correction2}")

                variance_ma = state["variance_ma"]
                if self.use_adabelief:
                    variance_ma_belief = state["variance_ma_belief"]

                # print(f"variance_ma, upper loop = {variance_ma}")

                # update the exp averages
                if self.use_adabelief:
                    grad_ma.mul_(beta1).add_(grad, alpha=1 - beta1)
                    grad_residual = grad - grad_ma
                    variance_ma_belief.mul_(beta2).addcmul(grad_residual, grad_residual, value=1 - beta2)
                # print(f"upper loop grad = {grad.shape}")
                variance_ma.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # print(f"variance_ma, grad adjusted")
                variance_ma_debiased = variance_ma / bias_correction2

                variance_ma_sum += variance_ma_debiased.sum()
                # print(f"variance_ma_sum = {variance_ma_sum}")
                # else: #madgrad

                # should we dupe variance_ma since stable is assuming adam style] variance?

                # stable wd
                # variance_ma_sum += grad_sum_sq.sum()

            # print(f"variance hat sum = {exp_avg_sq_hat_sum}")
            # Calculate the sqrt of the mean of all elements in exp_avg_sq_hat

            # we will run this first epoch only and then memoize
        if not self.param_size:
            self.param_size = param_size
            print(f"params size saved")
            print(f"total param groups = {i+1}")
            print(f"total params in groups = {j+1}")

        if not self.param_size:
            raise ValueError("failed to set param size")

        # stable weight decay
        if self.use_madgrad:
            variance_normalized = torch.pow(variance_ma_sum / param_size, 1 / 3)
        else:
            variance_normalized = math.sqrt(variance_ma_sum / param_size)
        # variance_mean = variance_ma_sum / param_size
        if math.isnan(variance_normalized):
            raise RuntimeError("hit nan for variance_normalized")

        # debugging/logging
        if self.logging:
            self.tracking_variance_sum.append(variance_ma_sum.item())
            self.tracking_variance_normalized.append(variance_normalized)

        # print(f"variance_mean = {variance_mean}")
        # print(f"variance_normalized = {variance_normalized}")
        # else:
        #    variance_normalized = math.pow((variance_ma / self.param_size), .3333)

        # print(f"variance mean sqrt = {variance_normalized}")

        # phase 2 - apply weight decay and step
        # ===========================================
        for group in self.param_groups:
            # print(f"In second phase loop")
            step = state["step"]

            # Perform stable weight decay
            decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            momentum = group["momentum"]

            beta1, beta2 = group["betas"]

            # madgrad outer
            if self.use_madgrad:
                ck = 1 - momentum
                lamb = lr * math.pow(step, 0.5)

            # stable decay and / or norm loss
            # ==================================
            if decay:
                if not self.use_madgrad:
                    # stable weight decay
                    p.data.mul_(1 - decay * lr / variance_normalized)
                else:
                    p.data.mul_(1 - decay * lamb / variance_normalized)

            if self.normloss_active:
                # apply norm loss
                unorm = self.unit_norm(p.data)
                correction = 2 * self.normloss_factor * (1 - torch.div(1, unorm + self.eps))
                p.mul_(1 - lr * correction)

            # innner loop, params
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                inner_grad = p.grad

                if self.use_madgrad:
                    # ================== madgrad ============================
                    if "grad_sum_sq" not in state:
                        state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                        state["s"] = torch.zeros_like(p.data).detach()
                        if momentum != 0:
                            state["x0"] = torch.clone(p.data).detach()

                    if momentum != 0.0 and grad.is_sparse:
                        raise RuntimeError("momentum != 0 is not compatible with sparse gradients")

                    # centralize gradients
                    if self.use_gc:
                        inner_grad = centralize_gradient(
                            inner_grad,
                            gc_conv_only=self.gc_conv_only,
                        )

                    grad_sum_sq = state["grad_sum_sq"]
                    s = state["s"]
                    if momentum == 0:
                        # Compute x_0 from other known quantities
                        rms = grad_sum_sq.pow(1 / 3)
                        if self.softplus:
                            rms = F.softplus(rms, beta=self.beta_softplus)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    # Accumulate second moments

                    # print(f" grad = {grad}")
                    # print(f"lamb = {lamb}")
                    # print(f"gsumsq = {grad_sum_sq}")

                    grad_sum_sq.addcmul_(inner_grad, inner_grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3)
                    if self.softplus:
                        rms = F.softplus(rms, beta=self.beta_softplus)

                    # Update s
                    s.data.add_(inner_grad, alpha=lamb)

                    # Step
                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        # p is a moving average of z
                        p.data.mul_(1 - ck).add_(z, alpha=ck)

                else:  # adam with pnm core
                    # ============= adamW with pnm option ========================

                    grad = p.grad

                    beta1, beta2 = group["betas"]

                    grad_ma = state["grad_ma"]
                    variance_ma = state["variance_ma"]
                    if self.use_adabelief:
                        variance_ma_belief = state["variance_ma_belief"]

                    if self.momentum_pnm:

                        max_variance_ma = state["max_variance_ma"]

                        if state["step"] % 2 == 1:
                            grad_ma, neg_grad_ma = (
                                state["grad_ma"],
                                state["neg_grad_ma"],
                            )
                        else:
                            grad_ma, neg_grad_ma = (
                                state["neg_grad_ma"],
                                state["grad_ma"],
                            )

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    if self.momentum_pnm:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_variance_ma, variance_ma, out=variance_ma)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (variance_ma.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                    # centralize gradients
                    if self.use_gc:
                        grad = centralize_gradient(
                            grad,
                            gc_conv_only=self.gc_conv_only,
                        )
                    if self.use_gcnorm:
                        grad = normalize_gradient(grad)

                    if not self.use_adabelief:
                        grad_ma.mul_(beta1 ** 2).add_(grad, alpha=1 - beta1 ** 2)

                    noise_norm = math.sqrt((1 + beta2) ** 2 + beta2 ** 2)

                    step_size = lr / bias_correction1

                    # softplus the denom
                    if self.softplus:
                        denom = F.softplus(denom, beta=self.beta_softplus)

                    pnmomentum = (
                        grad_ma.mul(1 + self.momentum_pnm)
                        .add(neg_grad_ma, alpha=-self.momentum_pnm)
                        .mul(1 / noise_norm)
                    )

                    p.addcdiv_(pnmomentum, denom, value=-step_size)

                    # denom = variance_biased_ma.sqrt().add(eps)

                    # step_size = lr / bias_correction1

                    # update weights
                    # p.data.add_(weight_mod, alpha=-step_size)
                    # p.addcdiv_(grad_ma, denom, value=-step_size)
        # print(f"\n End optimizer step\n")

        # end of step processes....

        # lookahead
        # ---------------------
        if self.lookahead_active:
            self.lookahead_process_step()

        return loss

    #   Lookahead merge process
    def lookahead_process_step(self):
        """handles blending of params for lookahead step."""

        if not self.lookahead_active:
            return
        self.lookahead_step += 1

        if self.lookahead_step >= self.lookahead_mergetime:
            self.lookahead_step = 0
            # merge lookahead cached params and save current ones
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]

                    p.data.mul_(self.lookahead_alpha).add_(
                        param_state["lookahead_params"],
                        alpha=1.0 - self.lookahead_alpha,
                    )
                    # save for next merge
                    param_state["lookahead_params"].copy_(p.data)
