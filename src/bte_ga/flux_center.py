import itertools

import deepxde as dde
import numpy as np
from scipy.stats import gmean

from deeponet_mf import MultiFidelityModel

model = MultiFidelityModel()


def get_quadrature_points(path):
    def pores_to_coordinates(grid):
        N, L = 5, 2
        grid = np.array(grid)
        a = grid.reshape((N, N)).nonzero()
        points = []
        d = L / N
        for i, j in zip(*a):
            point = [-L / 2 + d / 2 + j * d, L / 2 - d / 2 - i * d]
            points.append(point)
        return np.array(points)

    path = pores_to_coordinates(path)
    quad_x, step = np.linspace(-1, 1, num=100, endpoint=False, retstep=True)
    quad_x += step / 2
    quad_x = np.array(list(itertools.product(quad_x, repeat=2)))
    d = np.array([np.linalg.norm(x - path, ord=np.inf, axis=1) for x in quad_x])
    return quad_x[np.amin(d, axis=1) < 0.1]


quad_x, step = np.linspace(-1, 1, num=100, endpoint=False, retstep=True)
quad_x += step / 2
# quad_w = step ** 2
quad_x = np.array(list(itertools.product(quad_x, repeat=2)))

center_quad_x = np.vstack(([0, 0], quad_x))
two_points_quad_x = np.vstack(([[-0.4, 0.4], [0.4, 0]], quad_x))
# path_quad_x = np.vstack(
#     (
#         np.array([[-40, 0], [-20, 0], [-20, 20], [0, 20], [20, 20], [20, 0], [40, 0]])
#         / 50,
#         quad_x,
#     )
# )
# path_v2_quad_x = np.vstack(
#     (
#         np.array([[-40, 0], [-20, 0], [-20, 20], [0, 20], [20, 20], [20, 0], [40, 0]])
#         / 50,
#         np.array(
#             [
#                 [-40, 40],
#                 [-20, 40],
#                 [0, 40],
#                 [20, 40],
#                 [40, 40],
#                 [-40, 20],
#                 [40, 20],
#                 [0, 0],
#                 [-40, -20],
#                 [-20, -20],
#                 [0, -20],
#                 [20, -20],
#                 [40, -20],
#                 [-40, -40],
#                 [-20, -40],
#                 [0, -40],
#                 [20, -40],
#                 [40, -40],
#             ]
#         )
#         / 50,
#         quad_x,
#     )
# )

# Desired path
# quad_x = get_quadrature_points([1] * 25)
# path = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# path_quad_x = get_quadrature_points(path)
# reverse_path = 1 - path
# reverse_path_quad_x = get_quadrature_points(reverse_path)


def validation():
    fname_test = "../data/bte/mf_test.npz"
    d = np.load(fname_test, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_test = d["y"]
    err = []
    for i in range(len(X_branch)):
        y_pred = model((X_branch[i], X_trunk[i] / 50))
        err.append(dde.metrics.l2_relative_error(y_test[i], y_pred))
    print(np.mean(err))

    for i in range(3):
        y_pred = model((X_branch[i], X_trunk[i] / 50))
        np.savetxt(f"test{i}.dat", np.hstack((X_trunk[i], y_test[i], y_pred)))


def interpolation():
    fname_test = "../data/bte/mf_test.npz"
    d = np.load(fname_test, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]

    print(X_branch[1][0])
    X = X_branch[1].astype(np.float32)
    for x in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        X[:, 7] = x
        print(X[0])
        y_pred = model((X, X_trunk[1] / 50))
        np.savetxt(f"interpolation_{x}.dat", np.hstack((X_trunk[1], y_pred)))


def entire_flux(X):
    g = dde.geometry.Rectangle([-50, -50], [50, 50])
    pt = g.uniform_points(10000)
    X = np.tile(X, (len(pt), 1))
    return pt, model((X, pt / 50))


def central_difference(h):
    """Return
    [[-h,  0, ..., 0]
     [ h,  0, ..., 0]
     [ 0, -h, ..., 0]
     [ 0,  h, ..., 0]
     ...
    ]
    """
    dx = np.zeros((25 * 2, 25))
    for i in range(25):
        dx[2 * i, i] = -h
        dx[2 * i + 1, i] = h
    return dx


def center_flux(X):
    """Returns the center flux for n geometry X.

    Args:
        X: 2D np.ndarray of shape (n, 25).

    Returns:
        1D np.ndarray of shape (n).
    """
    return np.ravel(model((X, np.zeros((len(X), 2)))))


def center_flux_grad_FDM(X, h2, dx):
    """Returns the center flux and its gradient for the geometry X.

    Args:
        X: 1D np.ndarray of shape (25).
    """
    X = np.vstack((X, X + dx))
    loc = np.zeros((len(X), 2))
    out = np.ravel(model((X, loc)))
    y = out[0]
    dy = (out[2::2] - out[1::2]) / h2
    return y, dy


def center_flux_grad_AD(X):
    """Returns the center flux and its gradient for n geometry X.

    Args:
        X: 2D np.ndarray of shape (n, 25).

    Returns:
        y: 1D np.ndarray of shape (n).
        dy: 2D np.ndarray of shape (n, 25).
    """
    y, dy = model.value_and_grad((X, np.zeros((len(X), 2))))
    return np.ravel(y), dy


def center_flux_normalized(X):
    """Returns the normalized center flux for a geometry X.

    Args:
        X: 2D np.ndarray of shape (25).
    """
    X = np.tile(X, (len(center_quad_x), 1))
    y = np.ravel(model((X, center_quad_x)))
    y_mean = np.mean(y[1:])
    return y[0] / y_mean


def center_flux_normalized_grad(X):
    """Returns the normalized center flux and its gradient for a geometry X.

    Args:
        X: 2D np.ndarray of shape (25).
    """
    X = np.tile(X, (len(center_quad_x), 1))
    y, dy = model.value_and_grad((X, center_quad_x))
    y_center = y[0, 0]
    y_sum = np.mean(y[1:, 0]) * 4
    dy_center = dy[0]
    dy_sum = np.mean(dy[1:], axis=0) * 4
    return (
        4 * y_center / y_sum,
        4 * (y_sum * dy_center - y_center * dy_sum) / y_sum ** 2,
    )


def two_points_flux_normalized(X):
    """Returns the normalized two-center flux for a geometry X.

    Args:
        X: 2D np.ndarray of shape (25).
    """
    if np.sum(X) > 11:
        return 0
    X = np.tile(X, (len(two_points_quad_x), 1))
    y = np.ravel(model((X, two_points_quad_x)))
    y_mean = np.mean(y[2:])
    return 0.5 * (y[0] + y[1]) / y_mean


def two_points_flux_normalized_grad(X):
    """Returns the normalized two-center flux and its gradient for a geometry X.

    Args:
        X: 2D np.ndarray of shape (25).
    """
    X = np.tile(X, (len(two_points_quad_x), 1))
    y, dy = model.value_and_grad((X, two_points_quad_x))
    y_center = (y[0, 0] + y[1, 0]) / 2
    y_sum = np.mean(y[2:, 0]) * 4
    dy_center = (dy[0] + dy[1]) / 2
    dy_sum = np.mean(dy[2:], axis=0) * 4
    return (
        4 * y_center / y_sum,
        4 * (y_sum * dy_center - y_center * dy_sum) / y_sum ** 2,
    )


def path_flux_normalized(X):
    """Returns the normalized two-center flux for a geometry X.

    Args:
        X: 2D np.ndarray of shape (25).
    """
    X = np.tile(X, (len(path_quad_x), 1))
    y = np.ravel(model((X, path_quad_x)))
    y_mean = np.mean(y[7:])
    return np.mean(y[:7]) / y_mean


def path_flux_normalized_grad(X):
    """Returns the normalized path flux and its gradient for a geometry X.

    Args:
        X: 2D np.ndarray of shape (25).
    """
    X = np.tile(X, (len(path_quad_x), 1))
    y, dy = model.value_and_grad((X, path_quad_x))
    y_center = np.mean(y[:7])
    y_sum = np.sum(y[7:, 0]) * quad_w  # midpoint rule
    dy_center = np.mean(dy[:7], axis=0)
    dy_sum = np.sum(dy[7:], axis=0) * quad_w  # midpoint rule
    return (
        4 * y_center / y_sum,
        4 * (y_sum * dy_center - y_center * dy_sum) / y_sum ** 2,
    )


def path_v2_flux_normalized(X):
    """Returns the normalized two-center flux for a geometry X.

    Args:
        X: 2D np.ndarray of shape (25).
    """
    if np.prod(1 - X[[6, 7, 8, 10, 11, 13, 14]]) < 1e-5:
        return 0
    X = np.tile(X, (len(path_quad_x), 1))
    y = np.ravel(model((X, path_quad_x)))
    y_mean = np.mean(y[25:])  # midpoint rule
    y = y[:25] / y_mean
    y = np.maximum(y, 0)
    return np.prod(y[:7]) * np.prod(1 - y[7:25])
    # return gmean(y[:7] ** 0.02) * gmean(np.maximum(1 - y[7:25], 0))


def path_v3_flux_normalized(X):
    if np.sum(X) >= 16:
        return 0
    return path_flux_normalized(X)


def path_v4_flux_normalized(X):
    # if np.sum(X) >= 16:
    #     return 0
    y = np.ravel(model((np.tile(X, (len(quad_x), 1)), quad_x)))
    y_path = np.ravel(model((np.tile(X, (len(path_quad_x), 1)), path_quad_x)))
    return np.sum(y_path) / np.sum(y)
    # return np.mean(y_path) / np.max(y)
    # y_reverse_path = np.ravel(model((np.tile(X, (len(reverse_path_quad_x), 1)), reverse_path_quad_x)))
    # return (np.mean(y_path) - np.mean(y_reverse_path)) / np.max(y)


def brute_force():
    X = itertools.product([0, 1], repeat=25)  # 2 ** 25 = 33,554,432
    X = np.array(list(X))

    X = X[np.random.choice(2 ** 25, size=2 ** 16, replace=False)]

    y = np.ravel(center_flux(X))
    idx = np.argsort(y)
    print(y[idx[:10]])
    print(X[idx[:10]])
    print(y[idx[-10:]])
    print(X[idx[-10:]])


def main():
    # validation()
    # interpolation()
    # brute_force()
    # return

    # X = np.zeros(25)
    # X = np.ones(25)
    X = np.array(
        [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    )

    # h = 1e-4  # Cannot be too small for numerical stability
    # dx = central_difference(h)
    # y, dy = center_flux_grad_FDM(X, 2 * h, dx)
    # print(y)
    # print(dy)

    # X = np.zeros((1, 25))
    # X = np.ones((1, 25))
    # y = center_flux(X)
    # print(y)
    # y, dy = center_flux_grad_AD(X)
    # print(y)
    # print(dy)

    # y = center_flux_normalized(X)
    # print(y)
    # y, dy = center_flux_normalized_grad(X)
    # print(y)
    # print(dy)

    y = two_points_flux_normalized(X)
    print(y)
    # y, dy = two_points_flux_normalized_grad(X)
    # print(y)
    # print(dy)

    # y = path_flux_normalized(X)
    # print(y)
    # y, dy = path_flux_normalized_grad(X)
    # print(y)
    # print(dy)

    # y = path_v2_flux_normalized(X)
    # print(y)

    # y = path_v3_flux_normalized(X)
    # print(y)

    # y = path_v4_flux_normalized(X)
    # print(y)

    # pt, y = entire_flux(X)
    # np.savetxt("flux.dat", np.hstack((pt, y)))


if __name__ == "__main__":
    main()
