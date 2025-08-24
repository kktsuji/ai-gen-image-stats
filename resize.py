import cv2
import glob


def resize_image(image_path, output_path, width, height):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = cv2.resize(image, (width, height))

    # Save the resized image
    cv2.imwrite(output_path, resized_image)


input_path_list = glob.glob("./data/val/ctcs-lora/*")
out_dir = "./data/val-resize/ctcs-lora/"

for input_path in input_path_list:
    print("Processing...", input_path)
    output_path = out_dir + input_path.split("/")[-1]
    resize_image(input_path, output_path, 40, 40)
