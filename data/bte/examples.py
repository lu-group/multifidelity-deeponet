import matplotlib.pyplot as plt
import numpy as np

from utils import read_data, plot_variable


name = "bte5x5_2iter_size10532"
# N is the # of samples of you want to load
N = 1e5
i = 0

bte = read_data(name, N)
print("Size:", len(bte))
print(bte[i].keys())
print("")
# Mesh
print("nodes:", bte[i]["nodes"].shape)
print("elems:", len(bte[i]["elems"]), len(bte[i]["elems"][0]))
print("")
# Pore
print("x:", bte[i]["x"].reshape((5, 5)))
print("No. of pores:", np.sum(bte[i]["x"]))
# print("No. of pores:", len(bte[i]["centers"]))
# print(bte[i]["centers"])
print("")
# Output
print("centroids:", bte[i]["centroids"].shape)
print(bte[i]["centroids"])
print("")
print("variables:", bte[i]["variables"].keys())
print("Flux BTE:", bte[i]["variables"]["Flux BTE"]["data"].shape)
print("")
print("intermediate:", bte[i]["intermediate"].keys())
print("Flux BTE:", bte[i]["intermediate"]["Flux BTE"].shape)

plt.plot(bte[i]["nodes"][:, 0], bte[i]["nodes"][:, 1], "o")
plt.plot(bte[i]["centroids"][:, 0], bte[i]["centroids"][:, 1], "o")

# This is for plotting variables.
# variables can be: 'Temperature Fourier', 'Flux Fourier', 'Temperature BTE', or 'Flux BTE'.
# if you pick either flux, then direction may be specified.
# It can be either 0 (x) or 1 (y). If no direction is given, then the magnitude is considered.
# plot_structure(bte[i], "Temperature BTE", direction=None)
# plot_structure(bte[i], "Temperature Fourier", direction=None)
# plot_structure(bte[i], "Flux BTE", direction=None)
# plot_structure(bte[i], "Flux Fourier", direction=None)

# plot_variable(bte[i], "Temperature BTE", direction=None)
# plot_variable(bte[i], "Temperature Fourier", direction=None)
plot_variable(bte[i], "Flux BTE", direction=None)
plot_variable(bte[i], "Flux Fourier", direction=None)
plot_variable(bte[i], "Flux BTE", direction=None, sol="intermediate")
plt.show()
