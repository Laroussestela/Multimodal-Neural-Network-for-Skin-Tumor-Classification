from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# optimizers 
earlystopping = EarlyStopping(
                              monitor = 'val_accuracy', 
                              mode = 'max',
                              patience = 20,
                              restore_best_weights=True,
                              verbose = 1)

# filepath = './best_weights.hdf5'
checkpoint    = ModelCheckpoint('./best_weights.hdf5', 
                                monitor = 'val_accuracy', 
                                mode = 'max',
                                save_best_only=True, 
                                verbose = 1)


callback_list = [earlystopping, checkpoint]

# image model compile
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

image_model.compile(optimizer=optimizer,
                              loss="categorical_crossentropy",
                              metrics=[accuracy, precision, recall, specificity, f1_score])


# multimodal model compile
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)

image_text_model.compile(optimizer=optimizer, 
                            loss="categorical_crossentropy", 
                            metrics=[accuracy, precision, recall, specificity, f1_score])


# model fit
EPOCHS = 500
image_text_model_history = image_text_model.fit([x_img_train, x_meta_train], y_train,
                            validation_data=([x_img_val, x_meta_val], y_val), 
                            callbacks=callback_list, 
                            epochs=EPOCHS,
                            verbose=1)
