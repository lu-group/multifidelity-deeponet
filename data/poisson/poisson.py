import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from spaces import GRF
from utils import timing


def solver(f, N):
    """u_xx = 20f, x \in [0, 1]
    u(0) = u(1) = 0
    """
    h = 1 / (N - 1)
    K = -2 * np.eye(N - 2) + np.eye(N - 2, k=1) + np.eye(N - 2, k=-1)
    b = h ** 2 * 20 * f[1:-1]
    u = np.linalg.solve(K, b)
    u = np.concatenate(([0], u, [0]))
    return u


def example():
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    m = 100

    features = space.random(1)
    sensors = np.linspace(0, 1, num=m)
    sensor_values = space.eval_u(features, sensors[:, None])
    y = solver(sensor_values[0], m)
    np.savetxt("poisson_high.dat", np.vstack((sensors, np.ravel(sensor_values), y)).T)

    m_low = 10
    x_low = np.linspace(0, 1, num=m_low)
    f_low = space.eval_u(features, x_low[:, None])
    y_low = solver(f_low[0], m_low)
    np.savetxt("poisson_low.dat", np.vstack((x_low, y_low)).T)


@timing
def gen_data():
    print("Generating operator data...", flush=True)
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    m = 100
    num = 100000

    features = space.random(num)
    sensors = np.linspace(0, 1, num=m)
    sensor_values = space.eval_u(features, sensors[:, None])

    x = []
    y = []
    for i in range(num):
        tmp = solver(sensor_values[i], m)
        idx = np.random.randint(0, m, size=1)
        x.append(sensors[idx])
        y.append(tmp[idx])
    x = np.array(x)
    y = np.array(y)

    m_low = 10
    x_low = np.linspace(0, 1, num=m_low)
    f_low = space.eval_u(features, x_low[:, None])
    y_low = []
    y_low_x = []
    for i in range(num):
        tmp = solver(f_low[i], m_low)
        tmp = interpolate.interp1d(x_low, tmp, copy=False, assume_sorted=True)
        y_low.append(tmp(sensors))
        y_low_x.append(tmp(x[i]))
    y_low = np.array(y_low)
    y_low_x = np.array(y_low_x)
    np.savez_compressed(
        "../data/train.npz", X0=sensor_values, X1=x, y=y, y_low=y_low, y_low_x=y_low_x
    )

    # for i in range(5):
    #     plt.figure()
    #     plt.plot(sensors, sensor_values[i], "k")
    #     plt.plot(x[i], y[i], "or")
    #     plt.plot(sensors, y_low[i], "b")
    #     plt.plot(x[i], y_low_x[i], "xb")
    # plt.show()


def main():
    # example()
    gen_data()


if __name__ == "__main__":
    main()
