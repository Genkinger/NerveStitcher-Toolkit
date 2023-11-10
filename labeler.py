import nervestitcher
import sys
import cv2
from os.path import basename

images = nervestitcher.load_images_in_directory(sys.argv[1])

artefact_labels = [0] * len(images)
print(artefact_labels)
i = 0
while i < len(images):
    print(f"Showing Image {i}")
    cv2.imshow("current_image", images[i])
    key = cv2.waitKey()
    if key == ord("a"):
        print("pressed A")
        artefact_labels[i] = 1
        i += 1
    elif key == ord("n"):
        print("pressed N")
        artefact_labels[i] = 0
        i += 1
    elif key == ord("f"):
        if i < len(images) - 1:
            i += 1
        continue
    elif key == ord("b"):
        if i > 0:
            i -= 1
        continue
    elif key == 27:
        print("Aborting")
        break
    elif key == ord("p"):
        print(artefact_labels[:10])
        continue

with open(f"./data/artefacts_{basename(sys.argv[1])}.py", "w+") as outfile:
    outfile.write(f"{basename(sys.argv[1])}_artefact_mask = [")
    for v in artefact_labels:
        outfile.write(str(v))
        outfile.write(",")
    outfile.write("]")
