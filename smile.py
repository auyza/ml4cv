import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator
# สร้างโมเดล CNN สำหรับตรวจจับรอยยิ้ม
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# โหลดข้อมูลและเตรียม Dataset
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = data_gen.flow_from_directory(
    'dataset/smile',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = data_gen.flow_from_directory(
    'dataset/smile',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# โหลด Haarcascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_smile(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    predictions = []
    true_labels = []
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64))  # ขนาดภาพที่โมเดลต้องการ
        face_roi = face_roi / 255.0  # Normalize
        face_roi = np.reshape(face_roi, (1, 64, 64, 1))
        
        pred = model.predict(face_roi)[0][0]
        label = 1 if pred > 0.5 else 0  # กำหนด threshold 0.5
        
        predictions.append(pred)
        true_labels.append(label)
        
        # แสดงผลลัพธ์
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        text = "Smiling" if label == 1 else "Not Smiling"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame, predictions, true_labels

# โหลดหรือเทรนโมเดล
model = build_model()

# ฝึกโมเดล
model.fit(train_generator, validation_data=val_generator, epochs=10)

# เปิดกล้อง
cap = cv2.VideoCapture(0)
y_true = []
y_pred = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, pred, true_labels = detect_smile(frame, model)
    y_true.extend(true_labels)
    y_pred.extend(pred)
    
    cv2.imshow('Smile Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# สร้าง Confusion Matrix
cm = confusion_matrix(y_true, [1 if p > 0.5 else 0 for p in y_pred])
print("Confusion Matrix:\n", cm)

# คำนวณ ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
