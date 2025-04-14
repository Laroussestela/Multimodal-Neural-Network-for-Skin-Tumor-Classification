from tensorflow.keras import Sequential, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dense, Dropout, BatchNormalization, Input

IMAGE_SIZE = [64, 64]

image_model = Sequential([
    Input(shape=(*IMAGE_SIZE, 3)),
    Conv2D(16, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(16, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(32, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(32, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(64, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(128, 3, activation='relu', padding='same'),
    Dropout(0.1),
    Conv2D(128, 3, activation='relu', padding='same'),
    MaxPool2D((2, 2)),

    Conv2D(256, 3, activation='relu', padding='same'),
    Conv2D(256, 3, activation='relu', padding='same', name = 'last_conv_layer'),
    MaxPool2D((2, 2)),

    Flatten(),
    Dropout(0.2),   # 0.4
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),   # 0.5
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),   # 0.3
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),   # 0.3
    Dense(7, activation='softmax')        
], name = "image_model")
