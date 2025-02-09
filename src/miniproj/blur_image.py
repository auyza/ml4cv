import cv2
import numpy as np
import os
import glob  # For reading multiple images

def blur_and_save_images(input_dir, output_dir, blur_percentage):
    """
    Blurs multiple images in a directory and saves them to a new directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory to save blurred images.
        blur_percentage (float): Percentage of blur (0.0-1.0).

    Returns:
        int: The number of images processed successfully.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use glob to read multiple image files (supports various extensions)
    image_paths = glob.glob(os.path.join(input_dir, "*")) # Reads all files

    success_count = 0
    for image_path in image_paths:
        try:
            # Check if it's an image file (basic check)
            if not os.path.isfile(image_path) or not any(image_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                print(f"Skipping non-image file: {image_path}")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image at {image_path}")
                continue

            height, width = img.shape[:2]

            kernel_size = int(min(height, width) * blur_percentage)
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size <= 1:
              kernel_size = 3

            blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

            # Create the output filename (preserving original name)
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)  # Split name and extension
            output_path = os.path.join(output_dir, f"{name}_blurred{ext}") # Add "_blurred"

            cv2.imwrite(output_path, blurred_img)
            # print(f"Blurred image saved to {output_path}")
            success_count += 1

        except Exception as e:
            print(f"An error occurred processing {image_path}: {e}")

    return success_count


# Example usage:
input_dir = "data/datasets/mini-proj/original/smile/"  # Directory containing your input images
output_dir = "data/datasets/mini-proj/blur5/smile/"  # Directory to save blurred images
blur_percentage = 0.05  # 30% blur
print("start .... ")
num_processed = blur_and_save_images(input_dir, output_dir, blur_percentage)

print(f"Processed {num_processed} images successfully.")