# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
lamda = 2.13
theta = 3
lapse_rate = 0.05
p_right_if_lapse = 0.5 

def calculate_p_right(ild_range, lamda, theta, lapse_rate, p_right_if_lapse):
    """Calculate probability of right choice given ILD and model parameters."""
    return lapse_rate * p_right_if_lapse + (1 - lapse_rate) * (1 / (1 + np.exp(-2 * theta * np.tanh(lamda * ild_range/17.37))))

# %%
ild_range = np.arange(-16,16,0.1)
p_right = calculate_p_right(ild_range, 0.13, 30, lapse_rate, p_right_if_lapse)
p_right_1 = calculate_p_right(ild_range,2.13, theta, lapse_rate, p_right_if_lapse)

plt.plot(ild_range, p_right, label='ipl')
plt.plot(ild_range, p_right_1, label='2')
plt.axhline(1, color='black', linestyle='--')
plt.axhline(0, color='black', linestyle='--')

plt.legend()
plt.show()