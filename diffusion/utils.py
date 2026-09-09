"""
Diffusion utilities for the MDNPIC diffusion classification network:
beta schedules, forward q_sample, reverse p_sample and the ancestral
sampling loop. All reverse functions receive the three-dimensional
conditional guidance priors (y_global, y_local, y_texture) predicted by the
pre-trained MGCG network (Eq. 5, 9-11 and 12-18 in the paper).
"""
import math

import torch


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta)
             for i in range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
             for t in range(num_timesteps)])
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule}")
    return betas


def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None):
    """
    Sample y_t ~ q(y_t | y_0) with the guidance-dependent prior mean.

    :param y_0_hat: prior mean at timestep T given by the granularity guidance
                    (fused prediction or one of the granularity priors).
    """
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat \
          + sqrt_one_minus_alpha_bar_t * noise
    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(model, x, y, priors, y_T_mean, t, alphas, one_minus_alphas_bar_sqrt):
    """
    Reverse diffusion process sampling -- one time step.

    :param y:       sampled y at time step t, y_t.
    :param priors:  (y_global, y_local, y_texture) guidance priors from MGCG.
    :param y_T_mean: mean of the prior distribution at timestep T.
    """
    y_0_hat = priors[0]
    device = next(model.parameters()).device
    z = torch.randn_like(y)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square())
    eps_theta = model(x, y, t, *priors).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t)
    # posterior mean
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean
    # posterior variance
    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1


# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(model, x, y, priors, y_T_mean, one_minus_alphas_bar_sqrt):
    device = next(model.parameters()).device
    t = torch.tensor([0]).to(device)  # corresponding to timestep 1
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(x, y, t, *priors).to(device).detach()
    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t)
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1


def y_0_reparam(model, x, y, priors, y_T_mean, t, one_minus_alphas_bar_sqrt):
    """
    Obtain y_0 reparameterization from q(y_t | y_0), in which the noise term
    is the eps_theta prediction.
    """
    device = next(model.parameters()).device
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(x, y, t, *priors).to(device).detach()
    y_0_reparam = 1 / sqrt_alpha_bar_t * (
            y - (1 - sqrt_alpha_bar_t) * y_T_mean - eps_theta * sqrt_one_minus_alpha_bar_t).to(device)
    return y_0_reparam


def p_sample_loop(model, x, priors, y_T_mean, n_steps, alphas,
                  one_minus_alphas_bar_sqrt, only_last_sample=False):
    """
    Ancestral sampling loop of the reverse diffusion process (Eq. 12-18).
    """
    num_t, y_p_seq = None, None
    device = next(model.parameters()).device
    z = torch.randn_like(y_T_mean).to(device)
    cur_y = z + y_T_mean  # sampled y_T
    if only_last_sample:
        num_t = 1
    else:
        y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):
        y_t = cur_y
        cur_y = p_sample(model, x, y_t, priors, y_T_mean, t, alphas,
                         one_minus_alphas_bar_sqrt)  # y_{t-1}
        if only_last_sample:
            num_t += 1
        else:
            y_p_seq.append(cur_y)
    if only_last_sample:
        assert num_t == n_steps
        y_0 = p_sample_t_1to0(model, x, cur_y, priors, y_T_mean,
                              one_minus_alphas_bar_sqrt)
        return y_0
    else:
        assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0(model, x, y_p_seq[-1], priors, y_T_mean,
                              one_minus_alphas_bar_sqrt)
        y_p_seq.append(y_0)
        return y_p_seq
