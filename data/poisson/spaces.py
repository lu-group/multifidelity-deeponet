import matplotlib.pyplot as plt
import numpy as np
from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp

import config


class GRF(object):
    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        p = ProcessPool(nodes=config.processes)
        res = p.map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors).T,
            ys,
        )
        return np.vstack(list(res))


def space_samples(space, T):
    features = space.random(100000)
    sensors = np.linspace(0, T, num=1000)
    u = space.eval_u(features, sensors[:, None])

    plt.plot(sensors, np.mean(u, axis=0), "k")
    plt.plot(sensors, np.std(u, axis=0), "k--")
    plt.plot(sensors, np.cov(u.T)[0], "k--")
    plt.plot(sensors, np.exp(-0.5 * sensors ** 2 / 0.2 ** 2))
    for ui in u[:3]:
        plt.plot(sensors, ui)
    plt.show()


def main():
    space = GRF(1, length_scale=0.2, N=1000, interp="cubic")
    space_samples(space, 1)


if __name__ == "__main__":
    main()
