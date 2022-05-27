import sys

import deepxde as dde
import numpy as np


def get_data(fname_train, fname_test, residual=False, stacktrunk=False):
    d = np.load(fname_train, allow_pickle=True)
    idx = None
    # idx = np.random.choice(len(d["X0"]), size=1000, replace=False)

    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_train = d["y"]
    y_low_x = d["y_low_x"]
    if idx is not None:
        X_branch = X_branch[idx]
        X_trunk = X_trunk[idx]
        y_train = y_train[idx]
        y_low_x = y_low_x[idx]
    X_branch = np.vstack(X_branch).astype(np.float32)
    X_trunk = np.vstack(X_trunk).astype(np.float32) / 100
    y_train = np.vstack(y_train).astype(np.float32)
    y_low_x = np.vstack(y_low_x).astype(np.float32)
    if stacktrunk:
        X_trunk = np.hstack((X_trunk, y_low_x))
    if residual:
        y_train -= y_low_x
    X_train = (X_branch, X_trunk)

    d = np.load(fname_test, allow_pickle=True)
    X_branch = np.vstack(d["X0"]).astype(np.float32)
    X_trunk = np.vstack(d["X1"]).astype(np.float32) / 100
    y_test = np.vstack(d["y"]).astype(np.float32)
    y_low_x = np.vstack(d["y_low_x"]).astype(np.float32)
    if stacktrunk:
        X_trunk = np.hstack((X_trunk, y_low_x))
    if residual:
        y_test -= y_low_x
    X_test = (X_branch, X_trunk)
    return dde.data.Triple(X_train, y_train, X_test, y_test)


def main():
    fname_train = "../data/bte/train.npz"
    fname_test = "../data/bte/test.npz"
    data = get_data(fname_train, fname_test, residual=True, stacktrunk=True)

    m = 25
    dim_x = 3
    width = 256
    net = dde.maps.DeepONet(
        [m, width, width, width, width],
        [dim_x, width, width, width, width],
        "relu",
        "Glorot normal",
    )
    net.apply_output_transform(lambda _, y: 0.1 * y)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001)
    losshistory, train_state = model.train(
        epochs=100000,
        batch_size=2 ** 16,  # ~73 solutions (~902 points per solution)
    )


if __name__ == "__main__":
    main()
