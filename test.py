import fusion
import nervestitcher
import numpy as np
import matplotlib.pyplot as plt
import image_transform as it
import code
import torch

images = nervestitcher.load_images_in_directory("/home/leah/Datasets/snippet")
images = nervestitcher.preprocess_images(images[43:])


# coordinates, scores, descriptors = fusion.generate_interest_point_data(images)
# fusion.save_interest_point_data("./data/snippet_IP_0.005.pkl", coordinates, scores, descriptors)

# coordinates, scores, descriptors = fusion.load_interest_point_data("./data/snippet_IP_0.005.pkl")


# adjacency = fusion.generate_matching_data_n_vs_n(images, coordinates, scores, descriptors)
# fusion.save_adjacency_matrix("./data/snippet_adj.pkl", adjacency)


def bodged_fusion_attempt():
    images = nervestitcher.load_images_in_directory("/home/leah/Datasets/snippet")
    images = nervestitcher.preprocess_images(images[43:])
    adjacency = fusion.load_adjacency_matrix("./data/snippet_adj.pkl")

    canvas = np.zeros((1000, 1000))
    weights = np.zeros((1000, 1000))
    weight_image = fusion.generate_cos2_weight_image(384, 384)

    xs, ys, c = [], [], []
    for i, value in enumerate(adjacency[0][43:]):
        if value is None:
            continue
        value = value.astype(np.float32)
        if np.count_nonzero(value > 384) > 0:
            continue

        fusion.add_image_at_offset(weights, weight_image, -value + np.array([500, 500]))
        fusion.add_image_at_offset(canvas, images[i] * weight_image, -value + np.array([500, 500]))
        xs.append(value[0])
        ys.append(value[1])
        c.append(i)

    weights[np.nonzero(weights == 0)] = 1
    # plt.plot(xs, ys, marker="+")
    # plt.imshow(canvas / weights, cmap="Greys_r")
    plt.imsave("./data/bodge.jpg", canvas / weights, cmap="Greys_r")
    plt.show()


# bodged_fusion_attempt()


#########


def generate_superpoint_testing_data(non_artefacted_images, movement_functions, area_scale_factor):
    artefacted_images = []
    for image, movement in zip(non_artefacted_images, movement_functions):
        in_height, in_width = image.shape
        artefact = it.apply_artefact(
            image, int(in_width * area_scale_factor), int(in_height * area_scale_factor), movement
        )
        artefacted_images.append(artefact)
    return artefacted_images


def generate_random_movement_functions(count, strength):
    funcs = []
    for i in range(count):

        def func(t):
            return np.sin((t + 0.3) * np.pi * 2) * strength, 0

        funcs.append(func)
    return funcs


def side_by_side(image_a, image_b, coords_a=None, coords_b=None):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image_a, cmap="Greys_r")
    if coords_a is not None:
        axes[0].scatter(coords_a[:, 0], coords_a[:, 1], s=4, c="magenta")
    axes[1].imshow(image_b, cmap="Greys_r")
    if coords_b is not None:
        axes[1].scatter(coords_b[:, 0], coords_b[:, 1], s=4, c="magenta")
    fig.show()
    plt.show()


def crop_image(image, area_scale_factor):
    in_height, in_width = image.shape
    width = int(in_width * area_scale_factor)
    height = int(in_height * area_scale_factor)
    offset_x = int((in_width - width) / 2)
    offset_y = int((in_height - height) / 2)
    return image[offset_y : offset_y + height, offset_x : offset_x + height]


eval_images = nervestitcher.load_images_in_directory("/home/leah/Datasets/eval")
eval_images = nervestitcher.preprocess_images(eval_images)
movement_functions = generate_random_movement_functions(len(eval_images), 0.3)
artefacted_images = generate_superpoint_testing_data(eval_images, movement_functions, 2.0 / 3.0)
cropped_images = [crop_image(img, 2.0 / 3.0) for img in eval_images]

data_original = []
data_artefacted = []
for original, artefacted in zip(cropped_images, artefacted_images):
    input_original = torch.from_numpy(original).float()[None][None]
    input_artefacted = torch.from_numpy(artefacted).float()[None][None]

    c_o, s_o, d_o = nervestitcher.superpoint(input_original)
    c_a, s_a, d_a = nervestitcher.superpoint(input_artefacted)
    data_original.append((c_o, s_o, d_o))
    data_artefacted.append((c_a, s_a, d_a))

avg_orig = []
orig_cnt = []
avg_artef = []
artef_cnt = []
for orig, artef in zip(data_original, data_artefacted):
    _, s_o, _ = orig
    _, s_a, _ = artef
    avg_orig.append(np.average(s_o[0]))
    orig_cnt.append(len(s_o[0]))
    avg_artef.append(np.average(s_a[0]))
    artef_cnt.append(len(s_a[0]))
    print(
        f"original: {np.average(s_o[0])} | {len(s_o[0])}, artefacted: {np.average(s_a[0])} | {len(s_a[0])}"
    )
print(f"AVG ORIG: {np.average(np.array(avg_orig))} | {np.average(np.array(orig_cnt))}")
print(f"AVG ARTEF: {np.average(np.array(avg_artef))} | {np.average(np.array(artef_cnt))}")

i = 3
side_by_side(
    cropped_images[i],
    artefacted_images[i],
    data_original[i][0][0].cpu().numpy(),
    data_artefacted[i][0][0].cpu().numpy(),
)
