import numpy as np

def erf_(x):
    """
    Approximate the error function using a numerical approximation (Abramowitz & Stegun formula).
    
    Parameters:
        x (float or np.ndarray): The input value(s) for the error function.
    
    Returns:
        float or np.ndarray: The approximated error function value(s).
    """
    # Constants for the approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    x = np.asarray(x)  # Convert input to NumPy array if it isn't already
    sign = np.sign(x)  # Get sign for element-wise handling
    x = np.abs(x)

    # Approximation formula
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def d_A_RT_SCALAR(a, t):
    """
    Calculate the standard PA probability density function for a scalar time value.

    Parameters:
        a (float): Scalar parameter.
        t (float): Time value (must be > 0).

    Returns:
        float: The computed pdf value (0 if t <= 0).
    """
    if t <= 0:
        return 0.0
    p = (1.0 / np.sqrt(2 * np.pi * (t**3))) * np.exp(-((1 - a * t)**2) / (2 * t))
    return p


def stupid_f_integral_SCALAR(v, vON, theta, t, tp):
    """
    Calculate the PA pdf after the v_A change via an integral expression for scalar inputs.

    Parameters:
        v (float): Scalar parameter.
        vON (float): Scalar parameter.
        theta (float): Scalar parameter.
        t (float): Time value.
        tp (float): A shifted time value.

    Returns:
        float: The evaluated integral expression.
    """
    a1 = 0.5 * (1 / t + 1 / tp)
    b1 = theta / t + (v - vON)
    c1 = -0.5 * (vON**2 * t - 2 * theta * vON + theta**2 / t + v**2 * tp)

    a2 = a1
    b2 = theta * (1 / t + 2 / tp) + (v - vON)
    c2 = -0.5 * (vON**2 * t - 2 * theta * vON + theta**2 / t + v**2 * tp + 4 * theta * v + 4 * theta**2 / tp) + 2 * v * theta

    F01 = 1.0 / (4 * np.pi * a1 * np.sqrt(tp * t**3))
    F02 = 1.0 / (4 * np.pi * a2 * np.sqrt(tp * t**3))

    if a1 < 0:
        print(f'a1 = {a1}')
        print(f't = {t}, tp = {tp}')
        raise ValueError("a1 must be positive.")
    T11 = b1**2 / (4 * a1)
    T12 = (2 * a1 * theta - b1) / (2 * np.sqrt(a1))
    T13 = theta * (b1 - theta * a1)

    T21 = b2**2 / (4 * a2)
    T22 = (2 * a2 * theta - b2) / (2 * np.sqrt(a2))
    T23 = theta * (b2 - theta * a2)

    I1 = F01 * (T12 * np.sqrt(np.pi) * np.exp(T11 + c1) * (erf_(T12) + 1) + np.exp(T13 + c1))
    I2 = F02 * (T22 * np.sqrt(np.pi) * np.exp(T21 + c2) * (erf_(T22) + 1) + np.exp(T23 + c2))

    STF = I1 - I2
    return STF


def PA_with_LEDON_2_SCALAR(t, v, vON, a, tfix, tled, delta_i, delta_m):
    """
    Compute the PA pdf by combining contributions before and after LED onset for scalar inputs.

    Parameters:
        t (float): Time value.
        v (float): Drift parameter before LED.
        vON (float): Drift parameter after LED onset.
        a (float): Decision bound.
        tfix (float): Fixation time.
        tled (float): LED time.
        delta_A (float): Delta parameter.

    Returns:
        float: The combined PA pdf value.
    """
    # Check if the time (with fixation) is before the LED onset.
    if (t + tfix) <= (tled + 1e-6):
        # Use the scalar version of d_A_RT: note that d_A_RT_SCALAR must be defined.
        result = d_A_RT_SCALAR(v * a, (t - (delta_i + delta_m) + tfix) / (a**2)) / (a**2)
    else:
        t_clip = np.clip(t + tfix - delta_m - tled, 1e-6, None)
        tp_clip = np.clip(tled + tfix - delta_i, 1e-6, None)
        result = stupid_f_integral_SCALAR(v, vON, a, t_clip, tp_clip)
    return result


def d_A_RT_VEC(a, t):
    """
    Calculate the standard PA probability density function (vectorized).

    Parameters:
        a (float): Scalar parameter.
        t (numpy.ndarray): Time values (must be > 0).

    Returns:
        numpy.ndarray: The computed pdf values (0 where t <= 0).
    """
    t = np.asarray(t)  # Ensure t is a NumPy array
    p = np.zeros_like(t)
    valid_indices = t > 0
    p[valid_indices] = (1.0 / np.sqrt(2 * np.pi * (t[valid_indices]**3))) * np.exp(-((1 - a * t[valid_indices])**2) / (2 * t[valid_indices]))
    return p

def stupid_f_integral_VEC(v, vON, theta, t, tp):
    """
    Calculate the PA pdf after the v_A change via an integral expression (vectorized).

    Parameters:
        v (float): Scalar parameter.
        vON (float): Scalar parameter.
        theta (float): Scalar parameter.
        t (numpy.ndarray): Time values.
        tp (numpy.ndarray): A shifted time values.

    Returns:
        numpy.ndarray: The evaluated integral expressions.
    """
    t = np.asarray(t)
    tp = np.asarray(tp)
    a1 = 0.5 * (1 / t + 1 / tp)
    b1 = theta / t + (v - vON)
    c1 = -0.5 * (vON**2 * t - 2 * theta * vON + theta**2 / t + v**2 * tp)

    a2 = a1
    b2 = theta * (1 / t + 2 / tp) + (v - vON)
    c2 = -0.5 * (vON**2 * t - 2 * theta * vON + theta**2 / t + v**2 * tp + 4 * theta * v + 4 * theta**2 / tp) + 2 * v * theta

    F01 = 1.0 / (4 * np.pi * a1 * np.sqrt(tp * t**3))
    F02 = 1.0 / (4 * np.pi * a2 * np.sqrt(tp * t**3))

    T11 = b1**2 / (4 * a1)
    T12 = (2 * a1 * theta - b1) / (2 * np.sqrt(a1))
    T13 = theta * (b1 - theta * a1)

    T21 = b2**2 / (4 * a2)
    T22 = (2 * a2 * theta - b2) / (2 * np.sqrt(a2))
    T23 = theta * (b2 - theta * a2)

    I1 = F01 * (T12 * np.sqrt(np.pi) * np.exp(T11 + c1) * (erf_(T12) + 1) + np.exp(T13 + c1))
    I2 = F02 * (T22 * np.sqrt(np.pi) * np.exp(T21 + c2) * (erf_(T22) + 1) + np.exp(T23 + c2))

    STF = I1 - I2
    return STF


def PA_with_LEDON_2_VEC(t, v, vON, a, tfix, tled, delta_i, delta_m):
    """
    Compute the PA pdf by combining contributions before and after LED onset (vectorized).

    Parameters:
        t (numpy.ndarray): Time values.
        v (float): Drift parameter before LED.
        vON (float): Drift parameter after LED onset.
        a (float): Decision bound.
        tfix (float): Fixation time.
        tled (float): LED time.
        delta_A (float): Delta parameter.

    Returns:
        numpy.ndarray: The combined PA pdf values.
    """
    t = np.asarray(t)
    result = np.zeros_like(t)
    
    before_led = (t + tfix) <= (tled + 1e-6)
    result[before_led] = d_A_RT_VEC(v * a, (t[before_led] - (delta_i + delta_m) + tfix) / (a**2)) / (a**2)
    
    # Compute the time difference for the post-LED condition and clip negatives to 0.001.
    t_post_led = t[~before_led] + tfix - delta_m - tled
    t_post_led = np.clip(t_post_led, 1e-6, None)  # Clip any value below 0.001 to 0.001
    
    tp_post_led = tled + tfix - delta_i
    tp_post_led = np.clip(tp_post_led, 1e-6, None)  # Clip any value below 0.001 to 0.001

    result[~before_led] = stupid_f_integral_VEC(v, vON, a, t_post_led, tp_post_led)


    return result
