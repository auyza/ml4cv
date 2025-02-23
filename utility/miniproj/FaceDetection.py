import cv2
import os
from tqdm import tqdm

class FaceDetection:
    output_size =  (224, 224)     
    #โหลด Haar Cascades สำหรับตรวจจับใบหน้า
     
    def __init__(self):
        # ... your initialization code ...
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        

    def detect_and_resize_face(self, image_path, save_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ ไม่สามารถโหลดรูปภาพ: {os.path.basename(image_path)}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # แปลงเป็นภาพขาวดำสำหรับการตรวจจับที่แม่นยำ
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print(f"🚫 ไม่พบใบหน้าในรูปภาพ: {os.path.basename(image_path)}")
            return

        # ตัดเฉพาะใบหน้าจากภาพแรกที่ตรวจจับได้
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            resized_face = cv2.resize(face_img, self.output_size)
            cv2.imwrite(save_path, resized_face)
            # print(f"Save image : {save_path}")
            return  # บันทึกใบหน้าแรกและออกจากลูป


    def detectFaceAndResize(self, base_input_path, input_folders, output_folder):
        for folder in input_folders:
            input_path = os.path.join(base_input_path, folder)
            output_path = os.path.join(output_folder, folder)
            os.makedirs(output_path, exist_ok=True)

            if not os.path.exists(input_path):
                print(f"❌ ไม่พบโฟลเดอร์: '{input_path}'")
                continue

            images = os.listdir(input_path)
            print(f"\n📂 กำลังตรวจจับใบหน้าและ resize ในโฟลเดอร์ '{folder}'...")
            runno = 0

            for img_name in tqdm(images):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                
                runno += 1
                # base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(img_name) 
                out_img_name = f"RS-{runno}{ext}"

                img_path = os.path.join(input_path, img_name)
                save_path = os.path.join(output_path, out_img_name)
                self.detect_and_resize_face(img_path, save_path)
        

