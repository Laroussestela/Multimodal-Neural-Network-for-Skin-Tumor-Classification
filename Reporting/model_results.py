import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

results = image_text_model.evaluate([x_img_test, x_meta_test], y_test)

print(f'Pérdida (Loss): {results[0]}')
print(f'Precisión (Accuracy): {results[1]}')
print(f'Precisión (Precision): {results[2]}')
print(f'Recall: {results[3]}')
print(f'Specificidad (Specificity): {results[4]}')
print(f'F1-Score: {results[5]}')

# predictions
y_pred = image_text_model.predict([x_img_test, x_meta_test])

y_pred_classes = tf.argmax(y_pred, axis=-1)
y_true_classes = tf.argmax(y_test, axis=-1)


def confusion_matrix():
  cm = confusion_matrix(y_true_classes, y_pred_classes)
  cm_df = pd.DataFrame(cm, index=[f"Class {i}" for i in range(7)], columns=[f"Class {i}" for i in range(7)])
  
  plt.figure(figsize=(8,6))
  sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.title('Matriz de Confusión')
  plt.xlabel('Clase Predicha')
  plt.ylabel('Clase Real')
  plt.show()

def model_history():
  epochs = range(1, len(image_text_model_history.history['loss']) + 1)

  plt.plot(epochs, image_text_model_history.history['loss'], 'y', label='Training loss')
  plt.plot(epochs, image_text_model_history.history['val_loss'], 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  
  plt.plot(epochs, image_text_model_history.history['accuracy'], 'y', label='Training acc')
  plt.plot(epochs, image_text_model_history.history['val_accuracy'], 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()
  
  plt.plot(epochs, image_text_model_history.history['precision'], 'y', label='Training acc')
  plt.plot(epochs, image_text_model_history.history['val_precision'], 'r', label='Validation acc')
  plt.title('Training and validation precision')
  plt.xlabel('Epochs')
  plt.ylabel('Precision')
  plt.legend()
  plt.show()
  
  plt.plot(epochs, image_text_model_history.history['recall'], 'y', label='Training acc')
  plt.plot(epochs, image_text_model_history.history['val_recall'], 'r', label='Validation acc')
  plt.title('Training and validation recall')
  plt.xlabel('Epochs')
  plt.ylabel('Recall')
  plt.legend()
  plt.show()
  
  plt.plot(epochs, image_text_model_history.history['specificity'], 'y', label='Training acc')
  plt.plot(epochs, image_text_model_history.history['val_specificity'], 'r', label='Validation acc')
  plt.title('Training and validation specificity')
  plt.xlabel('Epochs')
  plt.ylabel('Specificity')
  plt.legend()
  plt.show()
  
  plt.plot(epochs, image_text_model_history.history['f1_score'], 'y', label='Training acc')
  plt.plot(epochs, image_text_model_history.history['val_f1_score'], 'r', label='Validation acc')
  plt.title('Training and validation f1_score')
  plt.xlabel('Epochs')
  plt.ylabel('F1_score')
  plt.legend()
  plt.show()
