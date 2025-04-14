from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def image_text_model():
    tabular_input = Input(shape=(x_meta_train.shape[1],))
    x = Dense(64, activation='relu')(tabular_input)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    # x = Dropout(0.1)(x) #
    # x = Dense(7, activation='softmax')(x)

    image_input = Input(shape=(64, 64, 3))
    y = Conv2D(16, 3, activation='relu', padding='same')(image_input)
    y = Dropout(0.1)(y)
    y = Conv2D(16, 3, activation='relu', padding='same')(y)
    y = MaxPool2D((2, 2))(y)

    y = Conv2D(32, 3, activation='relu', padding='same')(y)
    y = Dropout(0.1)(y)
    y = Conv2D(32, 3, activation='relu', padding='same')(y)
    y = MaxPool2D((2, 2))(y)

    y = Conv2D(64, 3, activation='relu', padding='same')(y)
    y = Dropout(0.1)(y)
    y = Conv2D(64, 3, activation='relu', padding='same')(y)
    y = MaxPool2D((2, 2))(y)

    y = Conv2D(128, 3, activation='relu', padding='same')(y)
    y = Dropout(0.1)(y)
    y = Conv2D(128, 3, activation='relu', padding='same')(y)
    y = MaxPool2D((2, 2))(y)

    y = Conv2D(256, 3, activation='relu', padding='same')(y)
    y = Dropout(0.1)(y)
    y = Conv2D(256, 3, activation='relu', padding='same')(y)
    y = MaxPool2D((2, 2))(y)

    y = Flatten()(y)
    y = Dropout(0.1)(y)
    y = Dense(512, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.1)(y)
    y = Dense(128, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.1)(y)
    y = Dense(64, activation='relu')(y)
    # y = BatchNormalization()(y) #
    # y = Dropout(0.1)(y)
    # y = Dense(7, activation='softmax')(y)

    combined = concatenate([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.1)(z)
    z = Dense(32, activation='relu')(z)
    z = Dropout(0.1)(z)
    final_output = Dense(7, activation='softmax')(z)

    model = Model(inputs=[image_input, tabular_input], outputs=final_output)
    return model
