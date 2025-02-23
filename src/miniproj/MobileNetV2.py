import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2  # หรือโมเดลอื่นๆ ที่คุณต้องการ
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns #for better visualization

# 1. Define Data Paths
dataset_dir = 'data/datasets/mini-proj/' #root directory where the images are.
original_data_path = os.path.join(dataset_dir,"resize") #where all images are stored, in class folders.
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# 2. Define Image Parameters
img_width, img_height = 224, 224  # Adjust as needed
batch_size = 32

# 3. Create Train/Validation/Test Split

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

class_names = os.listdir(original_data_path)
for class_name in class_names:
    if class_name.startswith('.'): #skip hidden files
        continue
    class_dir = os.path.join(original_data_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    images = os.listdir(class_dir)
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    for image in train_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(train_dir, class_name, image))
    for image in val_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(validation_dir, class_name, image))
    for image in test_images:
        shutil.copy(os.path.join(class_dir, image), os.path.join(test_dir, class_name, image))

print("Data preparation and labeling complete!")
# print(f"train file : {train_files}")


# 4. กำหนด image size และ batch size
img_height, img_width = 224, 224  # ปรับขนาดรูปภาพตามความเหมาะสม
batch_size = 32

# 5. สร้าง data generator สำหรับ train, validation และ test set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # 'categorical' for multiple classes, 'binary' for 2 classes
)

# print(f"Number of batches in train generator: {len(train_generator)}")



val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


# 6. สร้าง model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze base model

# Get the class names from the generator
classes = list(train_generator.class_indices.keys())
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)  # Output layer
model = Model(inputs=base_model.input, outputs=predictions)

# 7. compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 8. train model
epochs = 20
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

H = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# 9. ประเมิน model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


#========
# evaluate the network
print("[INFO] evaluating network...")
y_true = test_generator.classes
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=classes))

# plot the training loss and accuracy
print("[INFO] plotting the training loss and accuracy...")
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()


# Get predictions on the test set
# y_true = test_generator.classes  # True labels
# y_pred = model.predict(test_generator)  # Predicted probabilities

# # Convert predicted probabilities to class labels
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Create confusion matrix
# cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 5. Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes) #use seaborn for better visual
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 10. บันทึก model
model.save("src/miniproj/model_save/MobileNetV2-AsiaSmile.hdf5")

print("Data preparation, labeling, and model training complete!")