import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay)
import seaborn as sns

import tensorflow as tf
from tensorflow.io import TFRecordWriter
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks  import (
    Callback,
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint
)
from tensorflow.keras.layers import (
    Layer,
    GlobalAveragePooling2D,
    Conv2D,
    MaxPool2D,
    Dense,
    Flatten,
    InputLayer,
    BatchNormalization,
    Input,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomContrast,
    RandomBrightness,
    Resizing,
    Rescaling
)
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import L2, L1
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.train import Feature, Features, Example, BytesList, Int64List


BATCH = 32
SIZE = 224
SEED = 42

EPOCHS = 20
LR = 0.001
FILTERS = 6
KERNEL = 3
STRIDES = 1
REGRATE = 0.0
POOL = 2
DORATE = 0.05
LABELS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
NLABELS = len(LABELS)
DENSE1 = 1024
DENSE2 = 128

train_directory = 'data/datasets/Flower/train'
test_directory = 'data/datasets/Flower/test'

train_dataset = image_dataset_from_directory(
    train_directory,
    labels='inferred',
    label_mode='categorical',
    # class_names=LABELS,
    color_mode='rgb',
    batch_size=BATCH,
    image_size=(SIZE, SIZE),
    shuffle=True,
    seed=SEED,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

# Found 9206 files belonging to 48 classes.

test_dataset = image_dataset_from_directory(
    test_directory,
    labels='inferred',
    label_mode='categorical',
    # class_names=LABELS,
    color_mode='rgb',
    batch_size=BATCH,
    image_size=(SIZE, SIZE),
    shuffle=True,
    seed=SEED
)

# Found 3090 files belonging to 48 classes.

data_augmentation = Sequential([
        # Resizing(224, 224),
        RandomRotation(factor=0.25),
        RandomFlip(mode='horizontal'),
        RandomContrast(factor=0.1),
        RandomBrightness(0.1)
    ],
    name="img_augmentation",
)

training_dataset = (
    train_dataset
    .map(lambda image, label: (data_augmentation(image), label))
    .prefetch(tf.data.AUTOTUNE)
)


testing_dataset = (
    test_dataset.prefetch(
        tf.data.AUTOTUNE
    )
)

# Building the Efficient TF Model

# transfer learning

backbone = tf.keras.applications.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(SIZE, SIZE, 3),
    include_preprocessing=True
)

backbone.trainable = False

efficient_model = tf.keras.Sequential([
    Input(shape=(SIZE, SIZE, 3)),
    data_augmentation,
    backbone,
    GlobalAveragePooling2D(),
    Dense(DENSE1, activation='relu'),
    BatchNormalization(),
    Dense(DENSE2, activation='relu'),
    Dense(NLABELS, activation='softmax')
])

efficient_model.summary()

checkpoint_callback = ModelCheckpoint(
    '../best_weights',
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    save_best_only=True
)

loss_function = CategoricalCrossentropy()

metrics = [CategoricalAccuracy(name='accuracy')]

efficient_model.compile(
    optimizer = Adam(learning_rate=LR),
    loss = loss_function,
    metrics = metrics
)

# Model Training
efficient_history = efficient_model.fit(
    training_dataset,
    validation_data = testing_dataset,
    epochs = EPOCHS,
    verbose = 1
)

# loss: 0.2039
# accuracy: 0.9343
# val_loss: 0.3764
# val_accuracy: 0.9026