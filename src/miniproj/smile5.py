import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ฟังก์ชันสำหรับโหลดข้อมูลภาพจากโฟลเดอร์และเตรียมข้อมูลสำหรับการเรียนรู้

def load_training_data(data_dir):
    X, y = [], []
    for label in ['smiling', 'not_smiling']:
        folder_path = os.path.join(data_dir, label)
        if not os.path.exists(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))  # ปรับขนาดภาพให้เท่ากัน
            X.append(img.flatten())  # แปลงเป็นเวกเตอร์
            y.append(1 if label == 'smiling' else 0)
    return np.array(X), np.array(y)

# ฟังก์ชันสำหรับฝึกโมเดลและแสดงค่าที่ใช้สอน

def train_model():
    data_dir = "training_data"  # โฟลเดอร์ที่มีภาพสำหรับการฝึก
    X, y = load_training_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Data Size:", X_train.shape[0])
    print("Testing Data Size:", X_test.shape[0])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, "smile_detector.pkl")  # บันทึกโมเดล
    print("Model trained and saved as smile_detector.pkl")
    
    evaluate_model(model, X_test, y_test)

# ฟังก์ชันสำหรับประเมินผลโมเดลและแสดงรายงานผลลัพธ์

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Smiling', 'Smiling'], yticklabels=['Not Smiling', 'Smiling'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = "smile/confusion_matrix.png"
    plt.savefig(cm_path)  # บันทึก Confusion Matrix เป็นภาพ
    plt.show()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # แสดงผลรายงานการจำแนกประเภท
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_path = "smile/roc_curve.png"
    plt.savefig(roc_path)  # บันทึก ROC Curve เป็นภาพ
    plt.show()
    
    print(f"Confusion matrix saved at: {cm_path}")
    print(f"ROC curve saved at: {roc_path}")

#if __name__ == "__main1__":
    train_model()  # ฝึกโมเดลจากภาพที่เตรียมไว้

# ฟังก์ชันสำหรับตรวจจับรอยยิ้ม
# ใช้ Haarcascade เพื่อระบุใบหน้าและรอยยิ้มจากภาพที่ได้จากกล้อง

def detect_smile():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    cap = cv2.VideoCapture(0)  # เปิดกล้อง
    y_true, y_pred = [], []  # รายการสำหรับเก็บค่าจริงและค่าทำนาย
    
    if not os.path.exists("smile"):
        os.makedirs("smile")  # สร้างโฟลเดอร์ "smile" หากยังไม่มี
    
    frame_count = 0
    while True:
        ret, frame = cap.read()  # อ่านภาพจากกล้อง
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # แปลงภาพเป็นระดับสีเทา
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # ตรวจจับใบหน้า
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)  # ตรวจจับรอยยิ้ม
            
            is_smiling = len(smiles) > 0  # ตรวจสอบว่ามีรอยยิ้มหรือไม่
            y_true.append(1 if is_smiling else 0)
            y_pred.append(1 if is_smiling else 0)
            
            if is_smiling:
                cv2.putText(frame, 'Smiling :)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # วาดกรอบสีเขียวหากยิ้ม
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # วาดกรอบสีน้ำเงินหากไม่ยิ้ม
        
        cv2.imshow('Smile Detector', frame)  # แสดงผลลัพธ์
        
        frame_path = f"smile/frame_{frame_count}.png"
        cv2.imwrite(frame_path, frame)  # บันทึกภาพในโฟลเดอร์ "smile"
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13:  # หยุดระบบเมื่อกด 'q' หรือ Enter
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    evaluate_model(y_true, y_pred)

# ฟังก์ชันสำหรับประเมินผลโมเดล

def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # คำนวณ Confusion Matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Smiling', 'Smiling'], yticklabels=['Not Smiling', 'Smiling'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = "smile/confusion_matrix.png"
    plt.savefig(cm_path)  # บันทึก Confusion Matrix เป็นภาพ
    plt.show()
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))  # แสดงผลรายงานการจำแนกประเภท
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_path = "smile/roc_curve.png"
    plt.savefig(roc_path)  # บันทึก ROC Curve เป็นภาพ
    plt.show()
    
    print(f"Confusion matrix saved at: {cm_path}")
    print(f"ROC curve saved at: {roc_path}")

if __name__ == "__main__":
    detect_smile()  # เรียกใช้ฟังก์ชันตรวจจับรอยยิ้ม