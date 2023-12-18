import cv2
import numpy as np
import os

def process_files(file1, file2):
    # Load RGB image
    rgb_image = cv2.imread(file1)

    # Resize RGB image to 510x510
    rgb_image = cv2.resize(rgb_image, (510, 510))

    # Crop the center of the image to 384x384
    height, width = rgb_image.shape[:2]
    start_x = (width - 384) // 2
    start_y = (height - 384) // 2
    cropped_image = rgb_image[start_y:start_y+384, start_x:start_x+384]

    # Load grayscale image
    grayscale_image = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)

    # Create color map
    colormap = cv2.COLORMAP_JET

    # Apply color map to grayscale image
    colored_image = cv2.applyColorMap(grayscale_image, colormap)
    if cropped_image is None or colored_image is None:
        print(f"Error loading images: {file1}, {file2}")
        return

    # Overlay colored image onto cropped RGB image
    overlaid_image = cv2.addWeighted(cropped_image, 0.5, colored_image, 0.5, 0)

    cv2.imwrite(f"results/{os.path.basename(file1)}", overlaid_image)
    return overlaid_image


attn_dir = "../output/2023-12-18-11:10:19/test_attn_maps"
image_dir = "../datasets/CUB/test"
def get_filenames(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

attn_files = get_filenames(attn_dir)
image_files = get_filenames(image_dir)
print(attn_files)
print(image_files)
for attn_file in attn_files:
    attn_base_name = os.path.basename(attn_file)
    for image_file in image_files:
        image_base_name = os.path.basename(image_file)
        if attn_base_name == image_base_name:
            print(attn_base_name, image_base_name)
            process_files(image_file, attn_file)
            break


# result_root = "../output/2023-12-18-11:10:19"
# attn_map = "../output/2023-12-18-11:10:19/test_attn_maps/American_Crow_0080_25220.jpg"
# # image_root = "../datasets/CUB/test/037.Acadian_Flycatcher"
# image = "../datasets/CUB/test/029.American_Crow/American_Crow_0080_25220.jpg"

# process_files(f"{image}", f"{attn_map}")
