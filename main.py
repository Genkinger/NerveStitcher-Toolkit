import nervestitcher as ns
import torch
import config
import code

images = ns.load_images_in_directory("/home/leah/Dataset/snippet")
images = ns.preprocess_images(images[:40])
# images = images[:40]
input = torch.from_numpy(images).float()[:, None]
coordinates, scores, descriptors = ns.superpoint(input)

a, b = 20, 21
idx0, idx1, score0, score1 = ns.superglue(
    input[a][None],
    coordinates[a][None],
    scores[a][None],
    descriptors[a][None],
    input[b][None],
    coordinates[b][None],
    scores[b][None],
    descriptors[b][None],
    config.MATCHING_THRESHOLD,
)
# code.interact(local=locals())

valid = idx0[0] > -1
kpts_a = coordinates[a][valid]
kpts_b = coordinates[b][idx0[0][valid]]

code.interact(local=locals())

# code.interact(local=locals())

# plt.imshow(images[idx], cmap="Greys_r", vmin=0, vmax=1)
# plt.scatter(coordinates[idx][:, 1], coordinates[idx][:, 0], c="red", s=scores[idx] * 20)
# plt.show()
