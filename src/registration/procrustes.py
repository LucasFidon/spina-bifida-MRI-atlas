"""
Solver for the generalized procruste problem with weighted and missing values
"""
import numpy as np
from scipy.optimize import least_squares

SUPPORTED_MODE = ['full', 'scaling']


def solve_least_squares(lmks, g, w, x0, mode='scaling'):
    """
    Solve:
    minimize F(x) = 0.5 * sum(rho(f_k(x)**2), i = 0, ..., K - 1)
    subject to lb <= x <= ub
    :param lmks: K x D nd array; landmarks position
    :param g: K x D nd array; barycentres
    :param w: K nd array; weights
    :param x0: D*(D+1) array; initialization
    :return: affine transformation
    """
    def mat_from(x):
        if mode=='scaling':
            mat = np.array([
                [x[0], 0, 0],
                [0, x[1], 0],
                [0, 0, x[2]],
            ]).astype(np.float64)
        elif mode=='full':
            mat = x.reshape((D, D))
        else:
            raise ArgumentError('Mode %s not supported in Procrustes' % mode)
        return mat

    def fun(x):
        """
        Compute the f_k.
        :param x: line by line parameters of the affine matrix
        :return: K array; vector of residuals
        """
        mat = mat_from(x)
        t = np.zeros(D)
        res = apply_affine(landmarks=floating, mat=mat, translation=t)
        res -= reference
        res = np.linalg.norm(res, axis=1)  # K,
        res *= np.sqrt(weights)
        return res

    D = lmks.shape[1]

    # Check parameters
    assert mode in SUPPORTED_MODE, 'Found mode %s. Should be one of %s' % (mode, str(SUPPORTED_MODE))
    if mode == 'scaling':
        assert x0.size == D
    elif mode == 'full':
        assert x0.size == D * D

    weights = w / np.sum(w)  # re-normalize

    # Center the two configurations
    floating = np.copy(g)
    g_center = np.sum(weights[:, None] * g, axis=0)
    floating -= g_center
    reference = np.copy(lmks)
    lmks_center = np.sum(weights[:, None] * lmks, axis=0)
    reference -= lmks_center

    # Compute the optimal linear transformation
    if mode == 'scaling':  # only bounds for the diagonal scaling factors
        bounds = np.array([0.5, 2])
    else:
        bounds = np.array([-2, 2])
    result = least_squares(fun, x0=x0, bounds=bounds, verbose=0)
    mat_linear = mat_from(result.x)

    # Compute the optimal translation
    t0 = np.zeros(D)
    t = lmks_center - apply_affine(g_center, mat=mat_linear, translation=t0)

    return mat_linear, t


def apply_affine(landmarks, mat, translation):
    """
    :param landmarks: K x D array
    :param affine: D*(D+1) array
    :return:
    """
    res = np.matmul(landmarks.astype(np.float64), mat.astype(np.float64).T)  # K,D
    res += translation.astype(np.float64)
    return res


def apply_inverse_affine(landmarks, mat, translation):
    mat_inv = np.linalg.inv(mat.astype(np.float64))
    res = np.matmul(landmarks.astype(np.float64) - translation.astype(np.float64), mat_inv.T)
    return res


class WeightedGeneralizedProcrustes:
    def __init__(self, landmarks, weights, mode='scaling'):
        """
        N: number of samples
        K: number of landmarks
        D: space dimension
        :param landmarks: N x K x D nd array; Nan values supported
        :param weights: N x K nd array
        """
        # Check parameters
        assert mode in SUPPORTED_MODE, 'Found mode %s. Should be one of %s' % (mode, str(SUPPORTED_MODE))
        self.mode = mode

        self.N = landmarks.shape[0]
        self.K = landmarks.shape[1]
        self.D = landmarks.shape[2]
        self.lmks = landmarks.astype(np.float64)
        self.w = weights.astype(np.float64)

        # Remove the landmarks that are always missing
        lmks_nan = np.isnan(self.lmks)
        all_nans_col = []
        for k in range(self.K):
            if np.sum(lmks_nan[:, k, :]) == self.N * self.D:
                all_nans_col.append(k)
        self.lmks = np.delete(self.lmks, obj=all_nans_col, axis=1)
        self.w = np.delete(self.w, obj=all_nans_col, axis=1)
        self.K = self.lmks.shape[1]

        # Set the weights to 0 for NaNs
        lmks_nan = np.isnan(self.lmks)
        for i in range(self.N):
            for k in range(self.K):
                if np.sum(lmks_nan[i,k,:]) > 0:
                    self.w[i, k] = 0
        # We can replace the NaNs with 0 because
        # they will have no contribution to the optimization
        self.lmks[lmks_nan] = 0

        # Normalize the weights so that the sum over the samples is 1
        # It means that we give the same total weight to each landmark
        # irrespective of the number of missing values
        self.w /= np.sum(self.w, axis=0)

    def solve(self, max_iter=10):
        def iter_affines():
            linear = []
            translation = []
            for i in range(self.N):
                lmks_i = self.x[i,:,:]
                lmks_i[np.isnan(lmks_i)] = 0
                # Set the initialization
                if self.mode == 'scaling':
                    x0 = np.array([self.linear[i,l,l] for l in range(self.D)])
                else:
                    x0 = self.linear[i,:,:].flatten()
                # Solve the least squares problem
                mat_i, t_i = solve_least_squares(
                    lmks=lmks_i,
                    g=self.g,
                    w=self.w[i,:],
                    x0=x0,
                )
                linear.append(mat_i)
                translation.append(t_i)
            linear = np.stack(linear, axis=0)
            translation = np.stack(translation, axis=0)
            return linear, translation

        def iter_barycentre():
            warped_lmks = []
            for i in range(self.N):
                l_i = apply_inverse_affine(self.x[i,:,:], self.linear[i,:,:], self.translation[i,:])
                warped_lmks.append(l_i)
            x = np.stack(warped_lmks, axis=0)
            prod = x * self.w[:, :, None]
            g = np.sum(prod, axis=0)
            # Impose contraints on g
            g -= np.mean(g, axis=0)
            g *= self.g_dispersion / np.linalg.norm(g)
            g += self.g_mass_center
            return g

        # Create the normalized lmks positions
        self.x = np.copy(self.lmks)  # N,K,D

        # Compute the barycentre of the landmarks position for the given weights
        prod = self.x * self.w[:, :, None]
        self.g = np.sum(prod, axis=0)  # K,D
        # Compute the center of mass of g for contraints
        self.g_mass_center = np.mean(self.g, axis=0)  # D,
        # Compute the dispersion of the landmarks of g for constraints
        self.g_dispersion = np.linalg.norm(self.g - self.g_mass_center)

        # Initialize the output linear transformations to identity
        self.linear = np.stack([np.eye(self.D)] * self.N, axis=0).astype(np.float64)
        self.translation = np.stack([np.zeros(self.D)] * self.N, axis=0).astype(np.float64)

        # Run iterations
        has_converged = False
        iter = 0
        while not(has_converged) and iter < max_iter:
            iter += 1
            print('\nProcrustes iter', iter)

            # Update the affines transform
            new_linear, new_translation = iter_affines()
            delta = np.linalg.norm(self.linear - new_linear)
            delta += np.linalg.norm(self.translation - new_translation)
            self.linear = np.copy(new_linear)
            self.translation = np.copy(new_translation)

            # Update the barycentre
            self.g = iter_barycentre()
            # print('Norm of g:', np.linalg.norm(self.g))

            print('Delta=%f' % delta)
