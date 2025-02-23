import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load the pre-trained Xception model (without the top classification layer)
base_model = keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,
    classifier_activation="softmax",
    name="resnet50",
)
# 2. Freeze the base model's layers (optional, but often recommended initially)
base_model.trainable = False  # All layers are initially frozen

# 3. Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
num_classes = 2
predictions = Dense(num_classes, activation='softmax')(x)  # Final classification layer (num_classes is your number of classes)

# 4. Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# 5. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Use appropriate loss and metrics

# 6. Data augmentation (crucial for fine-tuning and preventing overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # Handle edge cases during augmentation
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale validation data

batch_size = 32
train_generator = train_datagen.flow_from_directory(
    "data/datasets/mini-proj/original/smile",  # Path to your training data directory
    target_size=(64, 64),  # Match Xception's input size
    batch_size=batch_size,
    class_mode='categorical'  # 'categorical' for multi-class, 'binary' for two classes
)

validation_generator = val_datagen.flow_from_directory(
    "data/datasets/mini-proj/original/test",  # Path to your validation data directory
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical'
)

# 7. Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Adjust patience as needed

# 8. Train the model (initial training with frozen base)
epochs_initial = 10  # Adjust as needed
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs_initial,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# 9. Fine-tuning (unfreeze some layers of the base model)
base_model.trainable = True  # Unfreeze all layers (or selectively unfreeze some)

# Example: Unfreeze the last few convolutional blocks (adjust as needed)
for layer in base_model.layers[:115]:  # Adjust this number based on Xception's architecture
    layer.trainable = False

# Recompile the model (important after changing trainable layers)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Use a lower learning rate for fine-tuning
              loss='categorical_crossentropy', metrics=['accuracy'])

# 10. Continue training (fine-tuning)
epochs_fine_tuning = 10  # Adjust as needed
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs_fine_tuning,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping]
)

# 11. Save the trained model
model.save("model_save/fine_tuned_xception_model.h5")  # Or .keras for TensorFlow 2.x