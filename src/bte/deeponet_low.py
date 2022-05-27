import sys

import numpy as np

import deepxde as dde
from deepxde.backend import tf


def get_data(fname_train, fname_test):
    d = np.load(fname_train, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_train = d["y"]
    X_branch = np.vstack(X_branch).astype(np.float32)
    X_trunk = np.vstack(X_trunk).astype(np.float32) / 50
    y_train = np.vstack(y_train).astype(np.float32)
    X_train = (X_branch, X_trunk)

    d = np.load(fname_test, allow_pickle=True)
    X_branch = np.vstack(d["X0"]).astype(np.float32)
    X_trunk = np.vstack(d["X1"]).astype(np.float32) / 50
    y_test = np.vstack(d["y"]).astype(np.float32)
    X_test = (X_branch, X_trunk)
    return dde.data.Triple(X_train, y_train, X_test, y_test)


def test(model, fname_test):
    d = np.load(fname_test, allow_pickle=True)
    X_branch = d["X0"]
    X_trunk = d["X1"]
    y_test = d["y"]
    err = []
    for i in range(len(X_branch)):
        y_pred = model.predict((X_branch[i], X_trunk[i] / 50))
        err.append(dde.metrics.l2_relative_error(y_test[i], y_pred))
    print(np.mean(err))

    for i in range(3):
        y_pred = model.predict((X_branch[i], X_trunk[i] / 50))
        np.savetxt(f"test{i}.dat", np.hstack((X_trunk[i], y_test[i], y_pred)))


def periodic_phase(x):
    m = 16
    phi0 = np.linspace(0, np.pi, num=m, endpoint=False, dtype=np.float32)

    x *= np.pi
    x, y = x[:, :1], x[:, 1:]
    phi1 = tf.Variable(phi0, trainable=True)
    phi2 = tf.Variable(phi0, trainable=True)
    phi3 = tf.Variable(phi0, trainable=True)
    phi4 = tf.Variable(phi0, trainable=True)
    return tf.math.cos(tf.concat([x - phi1, y - phi2, x + y - phi3, x - y - phi4], 1))


class LowFidelityModel:
    def __init__(self):
        fname_train = "../data/bte/low_train.npz"
        fname_test = "../data/bte/low_test.npz"
        data = get_data(fname_train, fname_test)
        m = 25
        dim_x = 2
        width = 512

        g = tf.Graph()
        with g.as_default():
            net = dde.maps.DeepONet(
                [m, width, width, width],
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
        self.model.restore("model_low/model.ckpt-486000", verbose=1)

    def __call__(self, X):
        return self.model.predict(X)

    def value_and_grad(self, X):
        feed_dict = self.model.net.feed_dict(False, X)
        return self.model.sess.run(
            [self.model.outputs, self.grad_xi], feed_dict=feed_dict
        )


def main():
    fname_train = "../data/bte/low_train.npz"
    fname_test = "../data/bte/low_test.npz"
    data = get_data(fname_train, fname_test)

    m = 25
    dim_x = 2
    width = 512
    net = dde.maps.DeepONet(
        [m, width, width, width],
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
        epochs=500000,
        batch_size=2 ** 16,  # ~73 solutions (~902 points per solution)
        callbacks=[checkpointer],
    )
    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    dde.postprocessing.save_loss_history(losshistory, "loss.dat")
    test(model, fname_test)


if __name__ == "__main__":
    main()
