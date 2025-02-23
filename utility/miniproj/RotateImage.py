import cv2
import os
import glob
from tqdm import tqdm

class RotateImage:
    def rotate_and_save_images(self, image_path, output_path, angle):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image at {image_path}")

            height, width = img.shape[:2]
            center = (width // 2, height // 2)  # Center of the image

            # Create the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0) # 1.0 is the scale

            # Perform the rotation
            rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height)) # Keep original size

            # base_name = os.path.basename(image_path)
            # name, ext = os.path.splitext(base_name)
            # output_path = os.path.join(output_dir, f"{name}_rotated{ext}")

            cv2.imwrite(output_path, rotated_img)
            # print(f"Rotated image saved to {output_path}")

        except Exception as e:
            print(f"An error occurred processing {image_path}: {e}")

        return 1

    def rotateImage(self, base_input_path, input_folders, output_folder, rotation_angle):
        success_count= 0
        for folder in input_folders:
            input_path = os.path.join(base_input_path, folder)
            output_path = os.path.join(output_folder, folder)
            os.makedirs(output_path, exist_ok=True)

            if not os.path.exists(input_path):
                print(f"❌ ไม่พบโฟลเดอร์: '{input_path}'")
                continue

            images = os.listdir(input_path)
            print(f"\n📂 Rotate images '{folder}'...")
            runno = 0
            for img_name in tqdm(images):
                
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                
                runno += 1
                # base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(img_name) 
                out_img_name = f"Blur-{runno}{ext}" 
                img_path = os.path.join(input_path, img_name)
                save_path = os.path.join(output_path, out_img_name)
                # print(f"Image file : {save_path}")
                success_count += self.rotate_and_save_images(img_path, save_path, rotation_angle)
        
        return success_count