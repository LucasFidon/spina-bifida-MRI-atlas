import numpy as np


def gaussian_kernel(ga, target_ga, std):
    # we don't normalize
    w = np.exp(-0.5 * ((ga - target_ga) / std)**2)
    return w
