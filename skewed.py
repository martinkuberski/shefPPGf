from scipy.special import erf
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def skewed_gaussian(t, a, loc, scale, shape):
    return (a / (scale * np.sqrt(2 * np.pi))) * np.exp(-np.square(t - loc)/(2 * np.square(scale))) * (1 + erf(shape * (t - loc)/np.sqrt(2) * scale))

def skewed_gaussian4(t, a1, loc1, scale1, shape1, a2, loc2, scale2, shape2, a3, loc3, scale3, shape3, a4, loc4, scale4, shape4):
    return (skewed_gaussian(t, a1, loc1, scale1, shape1)
            + skewed_gaussian(t, a2, loc2, scale2, shape2)
            + skewed_gaussian(t, a3, loc3, scale3, shape3)
            + skewed_gaussian(t, a4, loc4, scale4, shape4))

def fit(pulse, initials=[0.05, 0.2, 1/8, 0.1, 0.05, 0.4, 1/8, 0.1, 0.05, 0.6, 1/8, 0.1, 0.05, 0.8, 1/8, 0.1]):
    time = np.arange(pulse.size)
    time = time / pulse.size
    opt, covar = curve_fit(skewed_gaussian4, time, pulse, p0=initials, maxfev=100000, bounds=(0, np.inf))
    return opt