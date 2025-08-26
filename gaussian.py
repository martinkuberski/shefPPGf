import numpy as np
from scipy.optimize import curve_fit, fmin
from scipy.integrate import quad

def gaussian(time, amp, mean, var):
    return amp * np.exp(-(time - mean) ** 2 / (2 * (var ** 2)))

def gaussians(t, a1, m1, v1, a2, m2, v2, a3, m3, v3, a4, m4, v4):
    return ((a1 * np.exp(-(t - m1) ** 2 / (2 * (v1 ** 2)))) +
             (a2 * np.exp(-(t - m2) ** 2 / (2 * (v2 ** 2)))) +
             (a3 * np.exp(-(t - m3) ** 2 / (2 * (v3 ** 2)))) +
             (a4 * np.exp(-(t - m4) ** 2 / (2 * (v4 ** 2)))))

def gaussians2(t, a1, m1, v1, a2, m2, v2):
    return ((a1 * np.exp(-(t - m1) ** 2 / (2 * (v1 ** 2)))) +
            (a2 * np.exp(-(t - m2) ** 2 / (2 * (v2 ** 2)))))

def gaussian_array(amp, mean, var):
    time = np.arange(0, 1, 0.01)
    return [gaussian(t, amp, mean, var) for t in time]

def get_initials(pulse, values):
    max = np.max(pulse)
    duration = len(pulse)
    a1 = max * values[0]
    m1 = duration * values[1]
    v1 = values[2]
    a2 = max * values[3]
    m2 = duration * values[4]
    v2 = values[5]
    a3 = max * values[6]
    m3 = duration * values[7]
    v3 = values[8]
    a4 = max * values[9]
    m4 = duration * values[10]
    v4 = values[11]
    return [a1, m1, v1, a2, m2, v2, a3, m3, v3, a4, m4, v4]

def find_gaussians(pulse, initials):
    time = np.arange(pulse.size)
    # Normalise time
    time = time / pulse.size
    opt, covar = curve_fit(gaussians, time, pulse, p0=initials, maxfev=100000, bounds=(0,np.inf))
    return opt

def augmentation_index(a1, m1, v1, a2, m2, v2, a3):
    sysMax = -fmin(lambda t, a1, m1, v1, a2, m2, v2: -gaussians2(t, a1, m1, v1, a2, m2, v2), 0, (a1, m1, v1, a2, m2, v2), full_output=True, disp=False)[1]
    return sysMax / a3

def reflection_index(a1, m1, v1, a2, m2, v2, a3, m3, v3):
    sys_integral = quad(gaussians2, -np.inf, np.inf, (a1, m1, v1, a2, m2, v2))[0]
    three_integral = quad(gaussian, -np.inf, np.inf, (a3, m3, v3))[0]
    return sys_integral - three_integral

def sys_dia(a1, m1, v1, a2, m2, v2, a3, m3, v3, a4, m4, v4):
    sys_integral = quad(gaussians2, -np.inf, np.inf, (a1, m1, v1, a2, m2, v2))[0]
    dia_integral = quad(gaussians2, -np.inf, np.inf, (a3, m3, v3, a4, m4, v4))[0]
    return sys_integral / dia_integral