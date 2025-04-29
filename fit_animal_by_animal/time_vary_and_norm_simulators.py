
import numpy as np
import os

########### time varying func ##############
def K0(y, h, a, b):
    # a*sin(2*pi/h*(y-b)) where (y-b)>=0 and (y-b)<h/4, else 0
    return a * np.sin(2 * np.pi/h * (y - b)) * ((y - b) >= 0) * ((y - b) < h/4)

def K1(y, h, a, b):
    # (1+a*sin(2*pi/h*(y-b))) for (y-b) in [h/4, h/2)
    return (1 + a * np.sin(2 * np.pi/h * (y - b))) * ((y - b) >= h/4) * ((y - b) < h/2)

def K2(y, h, a, b):
    # (1 - a*(y-b)/h * exp(1 - (y-b)/h)) for (y-b)>=0
    return (1 - a * (y - b) / h * np.exp(1 - (y - b)/h)) * ((y - b) >= 0)

def phi_t_fn(y, h1, a1, b1, h2, a2):
    # Combination of K0, K1, and K2 functions
    return K0(y, h1, 1 + a1, b1) + K1(y, h1, a1, b1) + K2(y, h2, a2, b1 + h1/2)

################# normalization and time vary ###############################
def psiam_tied_data_gen_wrapper_rate_norm_time_vary_fn(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, \
                                t_stim, rate_norm_l,iter_num, N_print, phi_params, dt):

    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_rate_norm_time_vary(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, \
                                                       t_stim, t_A_aff, t_E_aff, del_go, rate_norm_l, phi_params, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}

def simulate_psiam_tied_rate_norm_time_vary(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, \
                                  t_A_aff, t_E_aff, del_go, rate_norm_l, phi_params, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    # mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    # sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    lambda_ABL_term = (10 ** (rate_lambda * (1 - rate_norm_l) * ABL / 20))
    lambda_ILD_arg = rate_lambda * ILD / chi
    lambda_ILD_L_arg = rate_lambda * rate_norm_l * ILD / chi
    
    is_act = 0
    while True:
        AI += V_A*dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            phi_t = phi_t_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
            mu = (1/T_0) * lambda_ABL_term * (np.sinh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg)) * phi_t
            sigma = np.sqrt( (1/T_0) * lambda_ABL_term * ( np.cosh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg) ) * phi_t )

            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t
            break
        elif DV <= -theta:
            choice = -1; RT = t
            break
        
        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0 # see if both AI and EA hit 
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff: 
                    phi_t = phi_t_fn(max(t - t_stim - t_E_aff, 1e-6), phi_params.h1, phi_params.a1, phi_params.b1, phi_params.h2, phi_params.a2)
                    mu = (1/T_0) * lambda_ABL_term * (np.sinh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg)) * phi_t
                    sigma = np.sqrt( (1/T_0) * lambda_ABL_term * ( np.cosh(lambda_ILD_arg) / np.cosh(lambda_ILD_L_arg) ) * phi_t )
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

################# ONLY normalization, NO time vary ###############################
def psiam_tied_data_gen_wrapper_rate_norm_fn(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, del_go, \
                                t_stim, rate_norm_l,iter_num, N_print, dt):

    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, \
                                                       t_stim, t_A_aff, t_E_aff, del_go, rate_norm_l, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim}

def simulate_psiam_tied_rate_norm(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, \
                                  t_A_aff, t_E_aff, del_go, rate_norm_l, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    # mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    # sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
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
            both_AI_hit_and_EA_hit = 0 # see if both AI and EA hit 
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