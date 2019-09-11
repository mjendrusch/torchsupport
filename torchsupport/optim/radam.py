import math
import torch
from torch.optim import Optimizer

class RAdam(Optimizer):
    r"""Implements RAdam algorithm.

    Uses modifications proposed in `On the Variance of the Adaptive Learning Rate and Beyond`_.
    Drops the AMSgrad functionality for now
    Provides AdamW style true weight decay

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): false weight decay = L2 penalty (default: 0)
        true_weight_decay (float, optional): weight decay for AdamW (default: 0) 
        rho_delay (float, optional): threshold after which second order term is taken into account. 
            (min: 4., default: 5.)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _On the Variance of the Adaptive Learning Rate and beyond:
        https://arxiv.org/abs/1908.03265
    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, true_weight_decay=0, rho_delay=5.):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 3/5 < betas[1] < 1.0: # Adapted to ensure RAdam stability
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 4 <= rho_delay:
            raise ValueError('Invalid rho_delay {}'.format(rho_delay))
        rho_inf = 2 / (1-betas[1]) - 1
        assert rho_inf > 4, 'RAdam expects that the rho_inf value is greater 4'
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, rho_inf=rho_inf,
                        true_weight_decay=true_weight_decay,
                        rho_delay=rho_delay)
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Cheap exponentiation
                    state['beta_1_exp'] = 1
                    state['beta_2_exp'] = 1

                # Name caching
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                step = state['step']
                rho_inf = group['rho_inf']

                step += 1
                # Cheap exponentiation
                state['beta_1_exp'] *= beta1
                state['beta_2_exp'] *= beta2
                # Name caching
                beta1exp = state['beta_1_exp']
                beta2exp = state['beta_2_exp']

                bias_correction1 = 1 - beta1exp
                bias_correction2 = 1 - beta2exp
                rho_step = rho_inf - 2 * step * beta2exp / (1 - beta2exp)

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)               # m_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  # v_t

                if rho_step <= group['rho_delay']:
                    step_size = group['lr'] / bias_correction1
                    if group['true_weight_decay'] != 0:
                        p.data *= 1 - (step_size * group['true_weight_decay'])
                    p.data.add_(-step_size, exp_avg)
                else:
                    r_factor = math.sqrt(
                        ((rho_step - 4) * (rho_step - 2) * rho_inf)
                        /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_step)
                    )

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    step_size = group['lr'] * r_factor * math.sqrt(bias_correction2) / bias_correction1

                    if group['true_weight_decay'] != 0:
                        p.data *= 1 - (step_size * group['true_weight_decay'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

