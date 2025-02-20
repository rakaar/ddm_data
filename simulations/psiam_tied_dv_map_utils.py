import numpy as np
from scipy.special import erf
import os

import random
# simulation part


######################  NO L ############################
def psiam_tied_data_gen_wrapper_V_A_change_no_L(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, \
                                t_stim_and_led_tuple, new_V_A, iter_num, N_print, is_LED_trial, dt):
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    # random element from t_stim_and_led_tuple
    t_stim, t_led = t_stim_and_led_tuple

    # print after every N_print iterations
    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_V_A_change_no_L(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, is_LED_trial, t_led, new_V_A, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim, 't_led': t_led, 'is_LED_trial': is_LED_trial}

def simulate_psiam_tied_V_A_change_no_L(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, is_LED_trial, t_led , new_V_A, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    
    is_act = 0
    while True:
        if t >= t_led and is_LED_trial:
            V_A = new_V_A

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
            is_act = 1
            AI_hit_time = t
            # if t*dt > t_stim - t_motor:
            while t <= (AI_hit_time + t_E_aff):#  u can process evidence till stim plays
                if t > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        break
                    elif DV <= -theta:
                        DV = -theta
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time
        if DV > 0:
            choice = 1
        elif DV < 0:
            choice = -1
        else: # if DV is 0 because stim has not yet been played, then choose right/left randomly
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1       
    
    return choice, RT, is_act

################# random if proactive wins ###############################
def psiam_tied_data_gen_wrapper_V_A_change_no_L_V2(V_A, theta_A, ABL_arr, ILD_arr, rate_lambda, T_0, theta_E, Z_E, t_A_aff, t_E_aff, \
                                t_stim_and_led_tuple, new_V_A, iter_num, N_print, is_LED_trial, dt):
    """
    if proactive wins, choose randomly
    """
    ABL = random.choice(ABL_arr)
    ILD = random.choice(ILD_arr)
    
    # random element from t_stim_and_led_tuple
    t_stim, t_led = t_stim_and_led_tuple

    # print after every N_print iterations
    if iter_num % N_print == 0:
        print(f'os id: {os.getpid()}, In iter_num: {iter_num}, ABL: {ABL}, ILD: {ILD}, t_stim: {t_stim}')

    choice, rt, is_act = simulate_psiam_tied_V_A_change_no_L_V2(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, is_LED_trial, t_led, new_V_A, dt)
    return {'choice': choice, 'rt': rt, 'is_act': is_act ,'ABL': ABL, 'ILD': ILD, 't_stim': t_stim, 't_led': t_led, 'is_LED_trial': is_LED_trial}

def simulate_psiam_tied_V_A_change_no_L_V2(V_A, theta_A, ABL, ILD, rate_lambda, T_0, theta_E, Z_E, t_stim, t_A_aff, t_E_aff, is_LED_trial, t_led , new_V_A, dt):
    AI = 0; DV = Z_E; t = t_A_aff; dB = dt**0.5
    
    chi = 17.37; q_e = 1
    theta = theta_E * q_e
    mu = (2*q_e/T_0) * (10**(rate_lambda * ABL/20)) * np.sinh(rate_lambda * ILD/chi)
    sigma = np.sqrt( (2*(q_e**2)/T_0) * (10**(rate_lambda * ABL/20)) * np.cosh(rate_lambda * ILD/ chi) )
    
    is_act = 0
    while True:
        if t >= t_led and is_LED_trial:
            V_A = new_V_A

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
            # if t*dt > t_stim - t_motor:
            while t <= (AI_hit_time + t_E_aff):#  u can process evidence till stim plays
                if t > t_stim + t_E_aff: # Evid accum wil begin only after stim starts and afferent delay
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
        if RT < t_led:
            if DV > 0:
                choice = 1
            elif DV < 0:
                choice = -1
            else: # if DV is 0 because stim has not yet been played, then choose right/left randomly
                # if proactive wins, choose randomly
                randomly_choose_up = np.random.rand() >= 0.5
                if randomly_choose_up:
                    choice = 1
                else:
                    choice = -1
        else:
            # both_AI_hit_and_EA_hit = 0 # TEMP: to not care if evidence bound was hit during the delay
            if both_AI_hit_and_EA_hit != 0:
                choice = both_AI_hit_and_EA_hit
            else:
                # If proactive wins after LED, choose randomly
                randomly_choose_up = np.random.rand() >= 0.5
                if randomly_choose_up:
                    choice = 1
                else:
                    choice = -1
    
    return choice, RT, is_act