import cv2
import os
from tqdm import tqdm

#โหลด Haar Cascades สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_resize_face(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ ไม่สามารถโหลดรูปภาพ: {os.path.basename(image_path)}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # แปลงเป็นภาพขาวดำสำหรับการตรวจจับที่แม่นยำ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"🚫 ไม่พบใบหน้าในรูปภาพ: {os.path.basename(image_path)}")
        return

    # ตัดเฉพาะใบหน้าจากภาพแรกที่ตรวจจับได้
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, output_size)
        cv2.imwrite(save_path, resized_face)
        # print(f"Save image : {save_path}")
        return  # บันทึกใบหน้าแรกและออกจากลูป


def loadImg(base_input_path, input_folders, output_folder):
    for folder in input_folders:
        input_path = os.path.join(base_input_path, folder)
        output_path = os.path.join(output_folder, folder)
        os.makedirs(output_path, exist_ok=True)

        if not os.path.exists(input_path):
            print(f"❌ ไม่พบโฟลเดอร์: '{input_path}'")
            continue

        images = os.listdir(input_path)
        print(f"\n📂 กำลังตรวจจับใบหน้าและ resize ในโฟลเดอร์ '{folder}'...")

        for img_name in tqdm(images):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            img_path = os.path.join(input_path, img_name)
            save_path = os.path.join(output_path, img_name)
            detect_and_resize_face(img_path, save_path)

# Example usage:
input_folders = ['non-smile', 'smile'] 
base_input_path = "data/datasets/mini-proj/original/"  # Directory containing your input images
base_output_path = "data/datasets/mini-proj/resize/"  # Directory to save blurred images
blur_percentage = 0.05  # 30% blur
output_size = (224, 224) 
print("start .... ")
num_processed = loadImg(base_input_path, input_folders, base_output_path)
print(f"\n✅ เสร็จสิ้นการตรวจจับใบหน้าและ resize! : {num_processed}")