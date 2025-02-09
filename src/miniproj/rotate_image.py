import cv2
import os
import glob

def rotate_and_save_images(input_dir, output_dir, angle):
    """
    Rotates multiple images in a directory and saves them to a new directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory to save rotated images.
        angle (int): Rotation angle in degrees (positive for counter-clockwise, negative for clockwise).

    Returns:
        int: The number of images processed successfully.
    """

    os.makedirs(output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_dir, "*"))

    success_count = 0
    for image_path in image_paths:
        try:
            if not os.path.isfile(image_path) or not any(image_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                print(f"Skipping non-image file: {image_path}")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image at {image_path}")
                continue

            height, width = img.shape[:2]
            center = (width // 2, height // 2)  # Center of the image

            # Create the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # 1.0 is the scale

            # Perform the rotation
            rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height)) # Keep original size

            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}_rotated{ext}")

            cv2.imwrite(output_path, rotated_img)
            print(f"Rotated image saved to {output_path}")
            success_count += 1

        except Exception as e:
            print(f"An error occurred processing {image_path}: {e}")

    return success_count


# Example usage:
input_dir = "data/datasets/mini-proj/original/smile/"  # Directory containing your input images
output_dir = "data/datasets/mini-proj/rotated10/smile/"  # Directory to save blurred images
rotation_angle = 10  # Rotate 10 degrees counter-clockwise
print("start .... ")
num_processed = rotate_and_save_images(input_dir, output_dir, rotation_angle)

print(f"Processed {num_processed} images successfully.")