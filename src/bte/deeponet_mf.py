import sys

import deepxde as dde
import numpy as np
from deepxde.backend import tf

from deeponet_low import LowFidelityModel

model_low = LowFidelityModel()


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
    X_trunk = np.vstack(X_trunk).astype(np.float32) / 50
    y_train = np.vstack(y_train).astype(np.float32)
    y_low_x = np.vstack(y_low_x).astype(np.float32)
    if stacktrunk:
        X_trunk = np.hstack((X_trunk, y_low_x))
    if residual:
        y_train -= y_low_x
    X_train = (X_branch, X_trunk)

    d = np.load(fname_test, allow_pickle=True)
    X_branch = np.vstack(d["X0"]).astype(np.float32)
    X_trunk = np.vstack(d["X1"]).astype(np.float32) / 50
    y_test = np.vstack(d["y"]).astype(np.float32)
    # y_low_x = np.vstack(d["y_low_x"]).astype(np.float32)  # Exact
    y_low_x = model_low((X_branch, X_trunk))  # Model
    if stacktrunk:
        X_trunk = np.hstack((X_trunk, y_low_x))
    if residual:
        y_test -= y_low_x
    X_test = (X_branch, X_trunk)
    return dde.data.Triple(X_train, y_train, X_test, y_test)


def test(model, fname_test):
    d = np.load(fname_test, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_test = d["y"]
    # y_low_x = d["y_low_x"]  # Exact
    err = []
    for i in range(len(X_branch)):
        # y_pred = model.predict((X_branch[i], np.hstack((X_trunk[i] / 50, y_low_x[i]))))  # Exact
        # y_pred += y_low_x[i]  # Exact
        y_low_x = model_low((X_branch[i], X_trunk[i] / 50))  # Model
        y_pred = model.predict(
            (X_branch[i], np.hstack((X_trunk[i] / 50, y_low_x)))
        )  # Model
        y_pred += y_low_x  # Model
        err.append(dde.metrics.l2_relative_error(y_test[i], y_pred))
    print(np.mean(err))

    for i in range(3):
        # y_pred = model.predict((X_branch[i], np.hstack((X_trunk[i] / 50, y_low_x[i]))))  # Exact
        # y_pred += y_low_x[i]  # Exact
        y_low_x = model_low((X_branch[i], X_trunk[i] / 50))  # Model
        y_pred = model.predict(
            (X_branch[i], np.hstack((X_trunk[i] / 50, y_low_x)))
        )  # Model
        y_pred += y_low_x  # Model
        np.savetxt(f"test{i}.dat", np.hstack((X_trunk[i], y_test[i], y_pred)))


def periodic_phase(inputs):
    m = 16
    phi0 = np.linspace(0, np.pi, num=m, endpoint=False, dtype=np.float32)

    x, y = inputs[:, :1] * np.pi, inputs[:, 1:2] * np.pi
    phi1 = tf.Variable(phi0, trainable=True)
    phi2 = tf.Variable(phi0, trainable=True)
    phi3 = tf.Variable(phi0, trainable=True)
    phi4 = tf.Variable(phi0, trainable=True)
    xy = tf.math.cos(tf.concat([x - phi1, y - phi2, x + y - phi3, x - y - phi4], 1))
    return tf.concat([xy, inputs[:, 2:]], 1)


def pores_to_coordinates(grid, N=5, L=2):
    d = L / N
    i, j = grid.reshape((N, N)).nonzero()
    x = -L / 2 + d / 2 + j * d
    y = L / 2 - d / 2 - i * d
    return np.vstack((x, y)).T


class MultiFidelityModel:
    def __init__(self):
        fname_train = "../data/bte/mf_train.npz"
        fname_test = "../data/bte/mf_test.npz"
        data = get_data(fname_train, fname_test, residual=True, stacktrunk=True)
        m = 25
        dim_x = 3
        width = 512

        g = tf.Graph()
        with g.as_default():
            net = dde.maps.DeepONet(
                [m, width],
                [dim_x, width, width, width, width],
                "relu",
                "Glorot normal",
            )
            net.apply_feature_transform(periodic_phase)
            net.apply_output_transform(
                lambda _, y: y * np.std(data.train_y) + np.mean(data.train_y)
            )
            self.model = dde.Model(data, net)
            self.model.compile("adam", lr=0)
            self.grad_xi = dde.grad.jacobian(net.outputs, net.inputs[0])
            self.grad_low = dde.grad.jacobian(net.outputs, net.inputs[1], j=2)
        self.model.restore("model_high/model.ckpt-398000", verbose=1)

    def __call__(self, X):
        # Check if x is inside a pore
        in_pore = []
        for i in range(len(X[0])):
            pores = pores_to_coordinates(X[0][i])
            d = np.linalg.norm(X[1][i] - pores, ord=np.inf, axis=1)
            in_pore.append(np.amin(d) < 0.1)
        in_pore = np.array(in_pore)[:, None]
        not_in_pore = 1 - in_pore

        y_low_x = model_low(X)
        y_pred = self.model.predict((X[0], np.hstack((X[1], y_low_x))))
        y_pred += y_low_x
        return y_pred * not_in_pore

    def value_and_grad(self, X):
        y_low_x, grad_xi_low = model_low.value_and_grad(X)
        feed_dict = self.model.net.feed_dict(False, (X[0], np.hstack((X[1], y_low_x))))
        y_high, grad_xi_high, grad_low = self.model.sess.run(
            [self.model.outputs, self.grad_xi, self.grad_low], feed_dict=feed_dict
        )
        y_high += y_low_x
        grad_xi = grad_xi_high + grad_xi_low * (1 + grad_low)
        return y_high, grad_xi


def main():
    fname_train = "../data/bte/mf_train.npz"
    fname_test = "../data/bte/mf_test.npz"
    data = get_data(fname_train, fname_test, residual=True, stacktrunk=True)

    m = 25
    dim_x = 3
    width = 512
    net = dde.maps.DeepONet(
        [m, width],
        [dim_x, width, width, width, width],
        "relu",
        "Glorot normal",
    )
    net.apply_feature_transform(periodic_phase)
    net.apply_output_transform(
        lambda _, y: y * np.std(data.train_y) + np.mean(data.train_y)
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001)
    checkpointer = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", save_better_only=True, period=1000
    )
    losshistory, train_state = model.train(
        epochs=400000,
        batch_size=2 ** 16,  # ~73 solutions (~902 points per solution)
        callbacks=[checkpointer],
    )
    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    dde.postprocessing.save_loss_history(losshistory, "loss.dat")
    test(model, fname_test)


if __name__ == "__main__":
    main()
