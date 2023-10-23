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


def image_grid_with_coordinates(images, coordinates, rows, columns, scores, colors="magenta"):
    fig, ax = plt.subplots(rows, columns, squeeze=False)
    row, col = 0, 0
    for i, image in enumerate(images):
        color = colors[i] if type(colors) is list else colors
        ax[row % rows, col % columns].imshow(image, cmap="Greys_r")
        ax[row % rows, col % columns].scatter(
            coordinates[i][:, 0], coordinates[i][:, 1], s=scores, c=color
        )
        col += 1
        if col % columns == 0:
            row += 1
    return fig, ax


def image_grid(images, rows, columns):
    fig, ax = plt.subplots(rows, columns, squeeze=False)
    row, col = 0, 0
    for i, image in enumerate(images):
        axis = ax[row % rows, col % columns]
        axis.imshow(image, cmap="Greys_r")
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        col += 1
        if col % columns == 0:
            row += 1
    return fig, ax
