import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
import shutil
import random

# 1. Define Data Paths
dataset_dir = 'data/datasets/mini-proj/' #root directory where the images are.
original_data_path = os.path.join(dataset_dir,"process") #where all images are stored, in class folders.
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

# 4. Create Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'  # 'categorical' for multiple classes, 'binary' for 2 classes
)

validation_generator = validation_datagen.flow_from_directory(
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

# # 5. Build the Model (Example: Simple CNN)
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(len(os.listdir(train_dir)), activation='softmax') # number of classes
# ])

# # 6. Compile the Model
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # 7. Train the Model
# epochs = 10  # Adjust as needed
# history = model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=validation_generator
# )

# # 8. Evaluate the Model
# test_loss, test_acc = model.evaluate(test_generator)
# print(f'Test accuracy: {test_acc}')

# # 9. Save the model
# model.save('my_model.h5')