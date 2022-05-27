import deepxde as dde
import numpy as np

from utils import read_data


def gen_low():
    bte = read_data("bte5x5_2iter_size10532", 10532)
    input_branch = []
    input_trunk = []
    output = []
    for i in range(10000):
        pores = bte[i]["x"]
        x = bte[i]["centroids"]
        y = np.linalg.norm(bte[i]["intermediate"]["Flux BTE"], axis=1)
        n = len(x)
        input_branch.append(np.tile(pores, (n, 1)))
        input_trunk.append(x)
        output.append(y.reshape((n, 1)))
    input_branch = np.array(input_branch, dtype=object)
    input_trunk = np.array(input_trunk, dtype=object)
    output = np.array(output, dtype=object)
    np.savez_compressed("train.npz", X0=input_branch, X1=input_trunk, y=output)


def gen_mf():
    bte = read_data("bte5x5_2iter_size10532", 10532)

    input_branch = []
    input_trunk = []
    output = []
    output_low = []
    for i in range(1000):
        pores = bte[i]["x"]
        # pores = np.ravel(bte[i]["centers"])
        x = bte[i]["centroids"]
        # T = bte[i]["variables"]["Temperature BTE"]["data"]
        # T_low = bte[i]["variables"]["Temperature Fourier"]["data"]
        y = np.linalg.norm(bte[i]["variables"]["Flux BTE"]["data"], axis=1)
        # y_low = np.linalg.norm(bte[i]["variables"]["Flux Fourier"]["data"], axis=1)
        y_low = np.linalg.norm(bte[i]["intermediate"]["Flux BTE"], axis=1)
        n = len(x)
        input_branch.append(np.tile(pores, (n, 1)))
        input_trunk.append(x)
        output.append(y.reshape((n, 1)))
        output_low.append(y_low.reshape((n, 1)))
    input_branch = np.array(input_branch, dtype=object)
    input_trunk = np.array(input_trunk, dtype=object)
    output = np.array(output, dtype=object)
    output_low = np.array(output_low, dtype=object)

    print(dde.metrics.mean_squared_error(np.vstack(output), np.vstack(output_low)))
    np.savez_compressed(
        "train.npz", X0=input_branch, X1=input_trunk, y=output, y_low_x=output_low
    )


def main():
    # gen_low()
    gen_mf()


if __name__ == "__main__":
    main()
