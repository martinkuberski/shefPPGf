import numpy as np
from scipy.optimize import curve_fit, fmin
from scipy.integrate import quad

# This file contains functions to perform Gaussian decomposition

def gaussian(time, amp, mean, var):
    '''
    Calculates the value of a Gaussian function with given parameters at given time

    :param time: Time at which the Gaussian function should be evaluated
    :param amp: Amplitude of the Gaussian function
    :param mean: Mean of the Gaussian function
    :param var: Variance of the Gaussian function
    :return: Value of the Gaussian function at the given time
    '''
    return amp * np.exp(-(time - mean) ** 2 / (2 * (var ** 2)))

def gaussians(t, a1, m1, v1, a2, m2, v2, a3, m3, v3, a4, m4, v4):
    '''
    Calculates the value of a sum of four Gaussian functions with given parameters at given time

    :param t: Time at which the Gaussian function sum should be evaluated
    :param a1: Amplitude of the first Gaussian function
    :param m1: Mean of the first Gaussian function
    :param v1: Variance of the first Gaussian function
    :param a2: Amplitude of the second Gaussian function
    :param m2: Mean of the second Gaussian function
    :param v2: Variance of the second Gaussian function
    :param a3: Amplitude of the third Gaussian function
    :param m3: Mean of the third Gaussian function
    :param v3: Variance of the third Gaussian function
    :param a4: Amplitude of the fourth Gaussian function
    :param m4: Mean of the fourth Gaussian function
    :param v4: Variance of the fourth Gaussian function
    :return: Value of the sum of the four Gaussian functions at the given time
    '''
    return ((a1 * np.exp(-(t - m1) ** 2 / (2 * (v1 ** 2)))) +
             (a2 * np.exp(-(t - m2) ** 2 / (2 * (v2 ** 2)))) +
             (a3 * np.exp(-(t - m3) ** 2 / (2 * (v3 ** 2)))) +
             (a4 * np.exp(-(t - m4) ** 2 / (2 * (v4 ** 2)))))

def gaussians2(t, a1, m1, v1, a2, m2, v2):
    '''
    Calculates the value of a sum of two Gaussian functions with given parameters at given time

    :param t: Time at which the Gaussian function sum should be evaluated
    :param a1: Amplitude of the first Gaussian function
    :param m1: Mean of the first Gaussian function
    :param v1: Variance of the first Gaussian function
    :param a2: Amplitude of the second Gaussian function
    :param m2: Mean of the second Gaussian function
    :param v2: Variance of the second Gaussian function
    :return: Value of the sum of the two Gaussian functions at the given time
    '''
    return ((a1 * np.exp(-(t - m1) ** 2 / (2 * (v1 ** 2)))) +
            (a2 * np.exp(-(t - m2) ** 2 / (2 * (v2 ** 2)))))

def gaussian_array(amp, mean, var):
    '''
    Generates an array representing the Gaussian function with given parameters

    :param amp: Amplitude of the Gaussian function
    :param mean: Mean of the Gaussian function
    :param var: Variance of the Gaussian function
    :return: An array representing the Gaussian function
    '''
    time = np.arange(0, 1, 0.01)
    return [gaussian(t, amp, mean, var) for t in time]

def find_gaussians(pulse, initials):
    '''
    Performs decomposition of a pulse into four Gaussian function by curve fitting

    :param pulse: Pulse to be decomposed
    :param initials: Initial values of the parameters (amplitude, mean, standard deviation for 4 Gaussian functions in an array)
    :return: Parameters of the four Gaussian functions fitting the pulse
    '''
    time = np.arange(pulse.size)
    # Normalise time
    time = time / pulse.size
    opt, covar = curve_fit(gaussians, time, pulse, p0=initials, maxfev=100000, bounds=(0,np.inf))
    return opt

def augmentation_index(a1, m1, v1, a2, m2, v2, a3):
    '''
    Calculates augmentation index of a decomposed pulse

    :param a1: Amplitude of the first Gaussian function
    :param m1: Mean of the first Gaussian function
    :param v1: Variance of the first Gaussian function
    :param a2: Amplitude of the second Gaussian function
    :param m2: Mean of the second Gaussian function
    :param v2: Variance of the second Gaussian function
    :param a3: Amplitude of the third Gaussian function
    :return: The augmentation index of the decomposed pulse
    '''
    sysMax = -fmin(lambda t, a1, m1, v1, a2, m2, v2: -gaussians2(t, a1, m1, v1, a2, m2, v2), 0, (a1, m1, v1, a2, m2, v2), full_output=True, disp=False)[1]
    return sysMax / a3

def reflection_index(a1, m1, v1, a2, m2, v2, a3, m3, v3):
    '''
    Calculates reflection index of a decomposed pulse

    :param a1: Amplitude of the first Gaussian function
    :param m1: Mean of the first Gaussian function
    :param v1: Variance of the first Gaussian function
    :param a2: Amplitude of the second Gaussian function
    :param m2: Mean of the second Gaussian function
    :param v2: Variance of the second Gaussian function
    :param a3: Amplitude of the third Gaussian function
    :param m3: Mean of the third Gaussian function
    :param v3: Variance of the third Gaussian function
    :return: The reflection index of the decomposed pulse
    '''
    sys_integral = quad(gaussians2, -np.inf, np.inf, (a1, m1, v1, a2, m2, v2))[0]
    three_integral = quad(gaussian, -np.inf, np.inf, (a3, m3, v3))[0]
    return sys_integral - three_integral

def sys_dia(a1, m1, v1, a2, m2, v2, a3, m3, v3, a4, m4, v4):
    '''
    Calculates the ratio of areas under the Gaussian functions representing the systolic phase to diastolic phase

    :param a1: Amplitude of the first Gaussian function
    :param m1: Mean of the first Gaussian function
    :param v1: Variance of the first Gaussian function
    :param a2: Amplitude of the second Gaussian function
    :param m2: Mean of the second Gaussian function
    :param v2: Variance of the second Gaussian function
    :param a3: Amplitude of the third Gaussian function
    :param m3: Mean of the third Gaussian function
    :param v3: Variance of the third Gaussian function
    :param a4: Amplitude of the fourth Gaussian function
    :param m4: Mean of the fourth Gaussian function
    :param v4: Variance of the fourth Gaussian function
    :return: The ratio of areas under the Gaussian functions representing the systolic phase to diastolic phase
    '''
    sys_integral = quad(gaussians2, -np.inf, np.inf, (a1, m1, v1, a2, m2, v2))[0]
    dia_integral = quad(gaussians2, -np.inf, np.inf, (a3, m3, v3, a4, m4, v4))[0]
    return sys_integral / dia_integral