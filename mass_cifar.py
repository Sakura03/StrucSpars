import numpy as np

data = {
    "res56-mass-0.3":np.array([[6.980e7, 4.018e5, 94.24],
                                   [7.072E7, 4.069E5, 93.96],
                                   [7.465E7, 4.153E5, 94.36]]),
    #
    "res56-mass-0.4":np.array([[6.404e7, 3.439e5, 93.82],
                                   [5.971E7, 3.558E5, 93.96],
                                   [6.063, 3.544E5, 93.85]]),
    #
    "res56-mass-0.5":np.array([[4.805e7, 2.99e5, 93.61],
                                   [4.857E7, 2.973E5, 93.76],
                                   [4.778E7, 2.927E5, 93.73]]),
    #
    "res56-mass-0.6":np.array([[4.195e7, 2.359e5, 93.30],
                                   [4.097E7, 2.236E5, 93.61],
                                   [4.306E7, 2.428E5, 93.54]]),
    #
    "res56-mass-0.7":np.array([[3.356e7, 1.797e5, 93.26],
                                   [3.684E7, 1.802E5, 93.23]]),
    #
    "res56-slimming-0.4":np.array([[5.613E7, 4.163E5, 93.11],
                                [5.69E7, 4.12E5, 92.78],
                                [5.95E7, 4.19E5, 93.40]]),
    #
    "res110-mass-0.1":np.array([[1.633E8, 1.029E6, 94.76],
                                [1.583E8, 1.014E6, 94.70],
                                [1.649E8, 1.029E6, 94.74]]),
    #
    "res110-mass-0.2":np.array([[1.475E8, 9.044E5, 94.92],
                                [1.509E8, 9.131E5, 94.78],
                                [1.444E8, 9.191E5, 94.64]]),
    #
    "res110-mass-0.3":np.array([[1.315E8, 7.936E5, 94.58],
                                [1.235E8, 7.815E5, 94.72],
                                [1.192E8, 7.919E5, 94.76]]),
    #
    "res110-mass-0.4":np.array([[1.171E8, 6.839E5, 94.88],
                                [1.171E8, 6.839E5, 94.88],
                                [1.171E8, 6.839E5, 94.88]]),
    #
    
    
}

for k, v in data.items():
    print("{:}: Acc {:.3f}/{:.3f}, Param {:.4E}/{:.4E}, FLOPS {:.4E}/{:.4E}"\
        .format(k, v[:, 2].mean(),v[:, 2].std(), v[:, 1].mean(), v[:, 1].std(), v[:, 0].mean(), v[:, 0].std()))