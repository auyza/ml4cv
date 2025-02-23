import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load the pre-trained ResNet50 model (without the top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Adjust input shape if needed

# 2. Freeze the base model's layers (optional, but often recommended initially)
base_model.trainable = False  # All layers are initially frozen
num_classes = 10

# 3. Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer (adjust units as needed)
predictions = Dense(num_classes, activation='softmax')(x)  # Add the final classification layer (num_classes is the number of your classes)

# 4. Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# 5. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Use appropriate loss and metrics

# 6. Data augmentation (important for fine-tuning and preventing overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to between 0 and 1
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale validation data

train_generator = train_datagen.flow_from_directory(
    train_data_dir,  # Path to your training data directory
    target_size=(224, 224),  # Match input size of ResNet50
    batch_size=batch_size,
    class_mode='categorical'  # 'categorical' for multi-class, 'binary' for two classes
)

validation_generator = val_datagen.flow_from_directory(
    validation_data_dir,  # Path to your validation data directory
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)


# 7. Train the model (initial training with frozen base)
epochs_initial = 10  # Adjust as needed
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs_initial,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# 8. Fine-tuning (unfreeze some layers of the base model)
base_model.trainable = True  # Unfreeze all layers (or selectively unfreeze some)

# Example: Unfreeze the last few convolutional blocks (adjust as needed)
for layer in base_model.layers[:100]:  # Freeze the first 100 layers (adjust this number)
    layer.trainable = False

# Recompile the model (important after changing trainable layers)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Use a lower learning rate for fine-tuning
              loss='categorical_crossentropy', metrics=['accuracy'])

# 9. Continue training (fine-tuning)
epochs_fine_tuning = 10  # Adjust as needed
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs_fine_tuning,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# 10. Save the trained model
model.save("model_save/fine_tuned_resnet50_model.h5")  # Or .keras for TensorFlow 2.x