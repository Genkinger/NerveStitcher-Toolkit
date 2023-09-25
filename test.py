from image_transform import apply_artefact, apply_gaussian_noise, generate_checkerboard_image
import matplotlib.pyplot as plt
import numpy


def movement(t: float) -> (int, int):
    if t < 0.5:
        return numpy.sin(2 * numpy.pi * t * 0.8) * 0.2, 0
    return (1 - numpy.exp(-30 * (t - 0.5))), 0


checkerboard = generate_checkerboard_image(512, 512, 12)
# checkerboard = apply_gaussian_noise(0, 0.004, checkerboard)
output = apply_artefact(checkerboard, 384, 384, movement)

plt.imshow(output, cmap="Greys")
plt.show()
