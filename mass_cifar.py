import numpy as np

data = {
    "res56-mass-0.3":np.array([[6.980e7, 4.018e5, 94.24],
                                   [7.072E7, 4.069E5, 93.96],
                                   [7.465E7, 4.153E5, 94.36]]),
    #
    "res56-mass-0.4":np.array([[6.404e7, 3.439e5, 93.82],
                                   [5.971E7, 3.558E5, 93.96],
                                   [6.063, 3.544E5, 93.85]]),

}

for k, v in data.items():
    print("{:}: Acc {:.3f}/{:.3f}, Param {:.4E}/{:.4E}, FLOPS {:.4E}/{:.4E}"\
        .format(k, v[:, 2].mean(),v[:, 2].std(), v[:, 1].mean(), v[:, 1].std(), v[:, 0].mean(), v[:, 0].std()))