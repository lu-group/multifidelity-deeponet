import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import numpy as np
import gzip
import pickle


def read_data(flavor, N=1e5):
    data = {}
    try:
        with gzip.open(flavor + ".npz", "rb") as f:
            k = 0
            while True:
                k += 1
                tmp = pickle.load(f)
                tmp["centroids"] = np.mean(tmp["nodes"][tmp["elems"], :2], axis=1)
                # tmp["centers"] = np.array(tmp["x"].reshape([5, 5]).nonzero()).T * [
                #     -10,
                #     10,
                # ] + np.array([100 - 5 - 50, 5 - 50])
                # tmp["centers"][:, [0, 1]] = tmp["centers"][:, [1, 0]]
                data.update({len(data): tmp})
                if k == N:
                    return data
    except EOFError:
        pass
    return data


def plot_structure(data, name, direction=None):
    fig, (ax) = plt.subplots(1, 1)
    viridis = cm.get_cmap("viridis")

    variable = data["variables"][name]["data"]
    if variable.ndim > 1:
        if direction == None:
            variable = np.linalg.norm(variable, axis=1)
        else:
            variable = variable.T[direction]

    minv = min(variable)
    maxv = max(variable)

    patches = []
    for n, elem in enumerate(data["elems"]):
        color = viridis((variable[n] - minv) / (maxv - minv))
        pp = data["nodes"][elem][:, :2]
        patches.append(Polygon(pp, color=color))

    p = PatchCollection(patches, match_original=True)
    ax.add_collection(p)
    a = 50
    ax.set_xlim([-a, a])
    ax.set_ylim([-a, a])
    ax.set_aspect("equal")
    plt.axis("off")


def plot_variable(data, name, direction=None, sol="variables"):
    if sol == "variables":
        variable = data[sol][name]["data"]
    elif sol == "intermediate":
        variable = data[sol][name]
    if variable.ndim > 1:
        if direction == None:
            variable = np.linalg.norm(variable, axis=1)
        else:
            variable = variable.T[direction]
    centroids = data["centroids"]

    plt.figure()
    plt.scatter(centroids[:, 0], centroids[:, 1], c=variable)
    plt.colorbar()
