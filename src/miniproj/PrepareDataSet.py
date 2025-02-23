import cv2
import os
from utility.miniproj import FaceDetection
from utility.miniproj import BlurImage
from utility.miniproj import RotateImage

# Detect image form dataset and resize:
input_folders = ['non-smile', 'smile'] 
base_input_path = "data/datasets/mini-proj/original/"  # Directory containing your input images
base_output_path = "data/datasets/mini-proj/resize/"  # Directory to save blurred images

output_size = (224, 224) 
print("start .... ")
# FaceDetection.parepareDataSets(base_input_path, input_folders, base_output_path)
fd = FaceDetection.FaceDetection()
fd.detectFaceAndResize(base_input_path, input_folders, base_output_path)
print(f"\n✅ เสร็จสิ้นการตรวจจับใบหน้าและ resize!")

input_dir = "data/datasets/mini-proj/resize/"  # Directory containing your input images
output_dir = "data/datasets/mini-proj/resize/"  # Directory to save blurred images
blur_percentage = 0.05  # 30% blur
bl = BlurImage.BlurImage()
success_count = bl.processBlueImage(input_dir, input_folders, output_dir, blur_percentage)
print(f"\n✅ blur image total {success_count}")


# input_dir = "data/datasets/mini-proj/resize/"  # Directory containing your input images
# output_dir = "data/datasets/mini-proj/resize/"  # Directory to save blurred images
rotation_angle = 10  # Rotate 10 degrees counter-clockwise
print("start .... ")
r = RotateImage.RotateImage()
num_processed = r.rotateImage(input_dir, input_folders, output_dir, rotation_angle)
print(f"Rotated {num_processed} images successfully.")



