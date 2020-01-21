from ml.codingame.code_busters import *

import matplotlib.pyplot as plt


def show_territory(territory: Territory):
    plt.imshow(territory.heat)
    for i in range(territory.w):
        for j in range(territory.h):
            h = round(territory.heat[i, j] * 10) / 10
            plt.text(j, i, h, ha="center", va="center", color="w")
    plt.show()


def test_territory():
    territory = Territory()
    # show_territory(territory)
    print(territory.heat)


test_territory()
