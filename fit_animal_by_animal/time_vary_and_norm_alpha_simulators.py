import os

import numpy as np

from time_vary_and_norm_simulators import phi_t_fn
from time_vary_norm_alpha_utils import gamma_omega_alpha_fn


def evidence_mu_sigma_alpha(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha, is_norm=True):
    gamma, omega = gamma_omega_alpha_fn(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha, is_norm)
    mu = gamma * omega * theta_E
    sigma = np.sqrt(omega) * theta_E
    return mu, sigma


def psiam_tied_data_gen_wrapper_rate_norm_alpha_fn(
        V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go,
        t_stim, rate_norm_l, alpha, iter_num, N_print, dt):
    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_rate_norm_alpha(
        V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E,
        t_stim, t_A_aff, t_E_aff, del_go, rate_norm_l, alpha, dt
    )
    return {'choice': choice, 'rt': rt, 'is_act': is_act, 'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}


def simulate_psiam_tied_rate_norm_alpha(
        V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim,
        t_A_aff, t_E_aff, del_go, rate_norm_l, alpha, dt):
    AI = 0
    DV = Z_E
    t = t_A_aff
    dB = dt**0.5

    theta = theta_E
    mu, sigma = evidence_mu_sigma_alpha(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha)

    is_act = 0
    while True:
        AI += V_A * dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            DV += mu * dt + sigma * np.random.normal(0, dB)

        t += dt

        if DV >= theta:
            choice = 1
            RT = t
            break
        elif DV <= -theta:
            choice = -1
            RT = t
            break

        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff:
                    DV += mu * dt + sigma * np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        both_AI_hit_and_EA_hit = 1
                        break
                    elif DV <= -theta:
                        DV = -theta
                        both_AI_hit_and_EA_hit = -1
                        break
                t += dt

            break

    if is_act == 1:
        RT = AI_hit_time
        if both_AI_hit_and_EA_hit != 0:
            choice = both_AI_hit_and_EA_hit
        else:
            if np.random.rand() >= 0.5:
                choice = 1
            else:
                choice = -1

    return choice, RT, is_act


def psiam_tied_data_gen_wrapper_rate_norm_alpha_time_vary_fn(
        V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go,
        t_stim, rate_norm_l, alpha, iter_num, N_print, phi_params, dt):
    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_rate_norm_alpha_time_vary(
        V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E,
        t_stim, t_A_aff, t_E_aff, del_go, rate_norm_l, alpha, phi_params, dt
    )
    return {'choice': choice, 'rt': rt, 'is_act': is_act, 'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}


def simulate_psiam_tied_rate_norm_alpha_time_vary(
        V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim,
        t_A_aff, t_E_aff, del_go, rate_norm_l, alpha, phi_params, dt):
    AI = 0
    DV = Z_E
    t = t_A_aff
    dB = dt**0.5

    theta = theta_E
    base_mu, base_sigma = evidence_mu_sigma_alpha(ABL, ILD, rate_lambda, T_0, theta_E, rate_norm_l, alpha)

    is_act = 0
    while True:
        AI += V_A * dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            curr_phi = phi_t_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
            DV += base_mu * curr_phi * dt + base_sigma * np.sqrt(curr_phi) * np.random.normal(0, dB)

        t += dt

        if DV >= theta:
            choice = 1
            RT = t
            break
        elif DV <= -theta:
            choice = -1
            RT = t
            break

        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff:
                    curr_phi = phi_t_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
                    DV += base_mu * curr_phi * dt + base_sigma * np.sqrt(curr_phi) * np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        both_AI_hit_and_EA_hit = 1
                        break
                    elif DV <= -theta:
                        DV = -theta
                        both_AI_hit_and_EA_hit = -1
                        break
                t += dt

            break

    if is_act == 1:
        RT = AI_hit_time
        if both_AI_hit_and_EA_hit != 0:
            choice = both_AI_hit_and_EA_hit
        else:
            if np.random.rand() >= 0.5:
                choice = 1
            else:
                choice = -1

    return choice, RT, is_act
