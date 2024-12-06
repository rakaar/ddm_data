import numpy as np
from scipy.special import erf
from scipy.integrate import quad, IntegrationWarning, quad_vec
import warnings

def rho_A_t_fn(t, V_A, theta_A):
    """
    For AI,prob density of t given V_A, theta_A
    """
    if t <= 0:
        return 0
    return (theta_A*1/np.sqrt(2*np.pi*(t)**3))*np.exp(-0.5 * (V_A**2) * (((t) - (theta_A/V_A))**2)/(t))


def P_t_x(x, t, v, a):
    """
    Prob that DV = x at time t given v, a 
    """
    if t <= 0:
        return 0
    return (1/np.sqrt(2 * (np.pi) * t)) * \
        ( np.exp(-((x - v*t)**2)/(2*t)) - np.exp( 2*v*a - ((x - 2*a - v*t)**2)/(2*t) ) )


def P_old_at_x_times_P_new_hit(x, t, V_A_old, V_A_new, a, t_LED):
    """
    Prob that DV is at x at t_LED and new V_A hits "a-x" bound at t - t_LED
    """
    return P_t_x(x, t_LED, V_A_old, a) * rho_A_t_fn(t-t_LED, V_A_new, a - x)



def PDF_hit_V_A_change(t, V_A_old, V_A_new, a, t_LED):
    """
    PDF of RT of hitting single bound with V_A change at t_LED
    """
    if t <= 0:
        return 0
    if t <= t_LED:
        p = rho_A_t_fn(t, V_A_old, a)
    else:
        p = quad(P_old_at_x_times_P_new_hit, -np.inf, a, args=(t, V_A_old, V_A_new, a, t_LED))[0]
    
    return p

def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def cum_A_t_fn(t, V_A, theta_A):
    """
    For AI, calculate cummulative distrn of a time t given V_A, theta_A
    """
    if t <= 0:
        return 0

    term1 = Phi(V_A * ((t) - (theta_A/V_A)) / np.sqrt(t))
    term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t) + (theta_A / V_A)) / np.sqrt(t))
    
    return term1 + term2

def P_old_at_x_times_CDF_new_hit(x, t, V_A_old, V_A_new, a, t_LED):
    """
    Prob that DV is at x at t_LED times CDF new V_A hits "a-x" bound at t - t_LED
    A helper func for CDF_hit_V_A_change
    """
    return P_t_x(x, t_LED, V_A_old, a) * cum_A_t_fn(t - t_LED, V_A_new, a - x)

def integrate_with_subranges(func, lower, upper, args, subrange_step):
        """
        Perform integration by dividing into subranges
        """
        subranges = np.arange(lower, upper, subrange_step)
        subranges = np.append(subranges, upper)  # Ensure upper bound is included
        total_integral = 0

        for i in range(len(subranges) - 1):
            sub_lower = subranges[i]
            sub_upper = subranges[i + 1]

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", IntegrationWarning)

                    # Perform integration over the subrange
                    result, _ = quad(func, sub_lower, sub_upper, args=args)
                    total_integral += result

                    # Check for warnings
                    if any(issubclass(warning.category, IntegrationWarning) for warning in w):
                        print(f"IntegrationWarning in subrange ({sub_lower}, {sub_upper}): "
                              f"Parameters - {args}")
            except Exception as e:
                print(f"Error in subrange ({sub_lower}, {sub_upper}): {e}")
                raise

        return total_integral


def CDF_hit_V_A_change(t, V_A_old, V_A_new, a, t_LED):
    """
    CDF of hitting bound with V_A change at t_LED
    """
    if t <= 0:
        return 0
    
    if t <= t_LED:
        return cum_A_t_fn(t, V_A_old, a)
    else:
        # return cum_A_t_fn(t_LED, V_A_old, a) + quad(P_old_at_x_times_CDF_new_hit, -np.inf, a, args=(t, V_A_old, V_A_new, a, t_LED))[0]
        # --- shift to numpy trapz
        a_pts = np.arange(-10, a, 0.01)
        func_values = np.array([P_old_at_x_times_CDF_new_hit(x, t, V_A_old, V_A_new, a, t_LED) for x in a_pts])
        integral_result = np.trapz(func_values, a_pts)
        return cum_A_t_fn(t_LED, V_A_old, a) + integral_result
        
        
        
        # --- shift to quad_vec that handles sharp discontinuity -----
        # return cum_A_t_fn(t_LED, V_A_old, a) + quad_vec(P_old_at_x_times_CDF_new_hit, -10, a, args=(t, V_A_old, V_A_new, a, t_LED))[0]

        # ---------- when warning print params and raise error -----
        # try:
        #     with warnings.catch_warnings(record=True) as w:
        #         warnings.simplefilter("always", IntegrationWarning)

        #         integral_result_1, _ = quad(
        #             P_old_at_x_times_CDF_new_hit,
        #             -10, 0,
        #             args=(t, V_A_old, V_A_new, a, t_LED)
        #         )

        #         integral_result_2, _ = quad(
        #             P_old_at_x_times_CDF_new_hit,
        #             0, a,
        #             args=(t, V_A_old, V_A_new, a, t_LED)
        #         )

        #         # Check for warnings in the first attempt
        #         if any(issubclass(warning.category, IntegrationWarning) for warning in w):
        #             print(f"IntegrationWarning with lower bound -10: Parameters causing issue - "
        #                   f"t={t}, V_A_old={V_A_old}, V_A_new={V_A_new}, a={a}, t_LED={t_LED}")
        #             raise RuntimeError(f"IntegrationWarning at lower bound -10: Parameters - "
        #                                f"t={t}, V_A_old={V_A_old}, V_A_new={V_A_new}, a={a}, t_LED={t_LED}")
            
        #     return cum_A_t_fn(t_LED, V_A_old, a) + integral_result_1 + integral_result_2

        # except Exception as e_second:
        #     print(f"Error during integration with lower bound 0: {e_second}")
        #     raise e_second

        # ---------- Change bound when warns -----
        # try:
        #     with warnings.catch_warnings(record=True) as w:
        #         warnings.simplefilter("always", IntegrationWarning)

        #         # First attempt: integrate with lower bound -10
        #         integral_result, _ = quad(
        #             P_old_at_x_times_CDF_new_hit,
        #             -10, a,
        #             args=(t, V_A_old, V_A_new, a, t_LED)
        #         )

        #         # Check for warnings in the first attempt
        #         if any(issubclass(warning.category, IntegrationWarning) for warning in w):
        #             print('-10 -> 0')                    
        #             # Retry with adjusted lower bound 0
        #             with warnings.catch_warnings(record=True) as w_retry:
        #                 warnings.simplefilter("always", IntegrationWarning)

        #                 integral_result, _ = quad(
        #                     P_old_at_x_times_CDF_new_hit,
        #                     0, a,
        #                     args=(t, V_A_old, V_A_new, a, t_LED)
        #                 )

        #                 # Check for warnings in the second attempt
        #                 if any(issubclass(warning.category, IntegrationWarning) for warning in w_retry):
        #                     print(f"IntegrationWarning with lower bound 0: Parameters causing issue - "
        #                           f"t={t}, V_A_old={V_A_old}, V_A_new={V_A_new}, a={a}, t_LED={t_LED}")
        #                     raise RuntimeError(f"Integration failed even with lower bound 0: "
        #                                        f"Parameters - t={t}, V_A_old={V_A_old}, V_A_new={V_A_new}, a={a}, t_LED={t_LED}")

        #         return cum_A_t_fn(t_LED, V_A_old, a) + integral_result
        # except Exception as e:
        #     print(f"Error during integration: {e}")
        #     raise

        # ---------- Sub integral breaks ----- 
        # try:
        #     # Integrate P_old_at_x_times_CDF_new_hit over subranges
        #     integral_result = integrate_with_subranges(
        #         P_old_at_x_times_CDF_new_hit, -10, a, 
        #         args=(t, V_A_old, V_A_new, a, t_LED), 
        #         subrange_step=0.5  # Adjust step size as needed
        #     )
        #     return cum_A_t_fn(t_LED, V_A_old, a) + integral_result
        # except Exception as e:
        #     print(f"Error during integration: {e}")
        #     raise
