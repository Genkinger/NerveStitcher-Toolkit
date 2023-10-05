from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch


def visualize_matches(image_a, image_b, coordinates_a, coordinates_b):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image_a, cmap="Greys_r")
    axes[1].imshow(image_b, cmap="Greys_r")
    for ka, kb in zip(coordinates_a, coordinates_b):
        con = ConnectionPatch(ka, kb, "data", "data", axes[0], axes[1], color="g", linewidth=1)
        fig.add_artist(con)
    fig.show()
    plt.show()


def visualize_coordinates(image, coordinates, scores):
    plt.imshow(image, cmap="Greys_r")
    plt.scatter(coordinates[:, 0], coordinates[:, 1], s=scores * 5, c="g")
    plt.show()
