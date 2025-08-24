import cv2
import numpy as np
import glob


def _apply_gamma_correction(img, gamma=2.2):
    img_normalized = img.astype(np.float32) / 255.0
    img_corrected = np.power(img_normalized, 1 / gamma)
    return (img_corrected * 255).astype(np.uint8)


if __name__ == "__main__":
    _COEFF = 1 / 2.2  # inverse gamma correction
    _HEIGHT = 40
    _WIDTH = 40

    input_path_list = glob.glob("./data/val/ctcs-lora/*")
    out_dir = "./data/val-resize/ctcs-lora/"

    for input_path in input_path_list:
        print("Processing...", input_path)
        output_path = out_dir + input_path.split("/")[-1]

        img = cv2.imread(input_path)
        img_gamma = _apply_gamma_correction(img, gamma=_COEFF)
        img_resized = cv2.resize(img_gamma, (_WIDTH, _HEIGHT))
        cv2.imwrite(output_path, img_resized)
