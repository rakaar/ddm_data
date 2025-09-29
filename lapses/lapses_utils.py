import numpy as np

def simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim,
                                  t_A_aff, t_E_aff, del_go, rate_norm_l, dt, lapse_prob=0.0, T_lapse_max=1.0):

    # Lapse mechanism: with probability lapse_prob, generate random choice and RT
    if np.random.rand() < lapse_prob:
        choice = 1 if np.random.rand() >= 0.5 else -1
        rt = t_stim + np.random.uniform(0, T_lapse_max)  # Uniform distribution between 0 and T_lapse_max
        is_act = 1  # Mark as lapse
        return choice, rt, is_act

    # Normal simulation process (with probability 1 - lapse_prob)
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5

    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    lambda_ABL_term = (10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20))
    lambda_ILD_arg = rate_lambda * ILD / chi
    lambda_ILD_L_arg = rate_lambda * rate_norm_l * ILD / chi
    mu = (1/T_0) * lambda_ABL_term * (np.sinh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg))
    sigma = np.sqrt( (1/T_0) * lambda_ABL_term * ( np.cosh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg) ) )

    is_act = 0
    while True:
        AI += V_A*dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)

        t += dt

        if DV >= theta:
            choice = +1; RT = t
            break
        elif DV <= -theta:
            choice = -1; RT = t
            break

        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff:
                    DV += mu*dt + sigma*np.random.normal(0, dB)
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
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1

    return choice, RT, is_act