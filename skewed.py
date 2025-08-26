from scipy.special import erf
from scipy.optimize import curve_fit
import numpy as np

# This file contains functions to perform skewed Gaussian decomposition

def skewed_gaussian(t, a, loc, scale, shape):
    '''
    Calculates the value of a skewed Gaussian function with given parameters at the given time

    :param t: Time to evaluate the skewed Gaussian function at
    :param a: Amplitude of the skewed Gaussian function
    :param loc: Location parameter of the skewed Gaussian function
    :param scale: Scale parameter of the skewed Gaussian function
    :param shape: Shape parameter of the skewed Gaussian function
    :return: The value of the skewed Gaussian function at the given time
    '''
    return (a / (scale * np.sqrt(2 * np.pi))) * np.exp(-np.square(t - loc)/(2 * np.square(scale))) * (1 + erf(shape * (t - loc)/np.sqrt(2) * scale))

def skewed_gaussian4(t, a1, loc1, scale1, shape1, a2, loc2, scale2, shape2, a3, loc3, scale3, shape3, a4, loc4, scale4, shape4):
    '''
    Calculates the value of a sum of four skewed Gaussian functions with given parameters at the given time

    :param t: Time to evaluate the sum of the skewed Gaussian functions at
    :param a1: Amplitude of the first skewed Gaussian function
    :param loc1: Location parameter of the first skewed Gaussian function
    :param scale1: Scale parameter of the first skewed Gaussian function
    :param shape1: Shape parameter of the first skewed Gaussian function
    :param a2: Amplitude of the second skewed Gaussian function
    :param loc2: Location parameter of the second skewed Gaussian function
    :param scale2: Scale parameter of the second skewed Gaussian function
    :param shape2: Shape parameter of the second skewed Gaussian function
    :param a3: Amplitude of the third skewed Gaussian function
    :param loc3: Location parameter of the third skewed Gaussian function
    :param scale3: Scale parameter of the third skewed Gaussian function
    :param shape3: Shape parameter of the third skewed Gaussian function
    :param a4: Amplitude of the fourth skewed Gaussian function
    :param loc4: Location parameter of the fourth skewed Gaussian function
    :param scale4: Scale parameter of the fourth skewed Gaussian function
    :param shape4: Shape parameter of the fourth skewed Gaussian function
    :return: The value of the skewed Gaussian function at the given time
    '''
    return (skewed_gaussian(t, a1, loc1, scale1, shape1)
            + skewed_gaussian(t, a2, loc2, scale2, shape2)
            + skewed_gaussian(t, a3, loc3, scale3, shape3)
            + skewed_gaussian(t, a4, loc4, scale4, shape4))

def fit(pulse, initials=[0.05, 0.2, 1/8, 0.1, 0.05, 0.4, 1/8, 0.1, 0.05, 0.6, 1/8, 0.1, 0.05, 0.8, 1/8, 0.1]):
    '''
    Performs decomposition of a PPG pulse into four Skewed Gaussian functions

    :param pulse: Pulse to be decomposed
    :param initials: Initial values of the parameters (amplitude, location, scale, and shape for 4 Gaussian functions)
    :return: Parameters of the four skewed Gaussian functions fitting the pulse
    '''
    time = np.arange(pulse.size)
    time = time / pulse.size
    opt, covar = curve_fit(skewed_gaussian4, time, pulse, p0=initials, maxfev=100000, bounds=(0, np.inf))
    return opt