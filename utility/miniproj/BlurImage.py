import cv2
import numpy as np
import os
import glob  # For reading multiple images
from tqdm import tqdm

class BlurImage:
    def __init__(self):
        pass
    
    def blur_and_save_images(self, image_path, output_path, blur_percentage):
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image at {image_path}")
                return 0

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
            # output_file = os.path.join(output_path, f"{name}_blurred{ext}") # Add "_blurred"

            cv2.imwrite(output_path, blurred_img)
            # print(f"Blurred image saved to {output_path}")

        except Exception as e:
            print(f"An error occurred processing {image_path}: {e}")
        return 1

    
    def processBlueImage(self, base_input_path, input_folders, output_folder, blur_percentage):
        success_count= 0
        for folder in input_folders:
            input_path = os.path.join(base_input_path, folder)
            output_path = os.path.join(output_folder, folder)
            os.makedirs(output_path, exist_ok=True)

            if not os.path.exists(input_path):
                print(f"❌ ไม่พบโฟลเดอร์: '{input_path}'")
                continue

            images = os.listdir(input_path)
            print(f"\n📂 กำลังตรวจจับ images '{folder}'...")

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
                success_count += self.blur_and_save_images(img_path, save_path, blur_percentage)
        
        return success_count


