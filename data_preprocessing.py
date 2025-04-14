import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from tqdm import tqdm
tqdm.pandas()
SIZE=64


skin_df = pd.read_csv('HAM10000_metadata.csv')

def balance_data()():
  le = LabelEncoder()
  le.fit(skin_df['dx'])
  
  skin_df['label'] = le.transform(skin_df['dx'])
  
  df_0 = skin_df[skin_df['label'] == 0]
  df_1 = skin_df[skin_df['label'] == 1]
  df_2 = skin_df[skin_df['label'] == 2]
  df_3 = skin_df[skin_df['label'] == 3]
  df_4 = skin_df[skin_df['label'] == 4]
  df_5 = skin_df[skin_df['label'] == 5]
  df_6 = skin_df[skin_df['label'] == 6]
  
  n_samples=1000 
  df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42) 
  df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42) 
  df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
  df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
  df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
  df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
  df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)
  
  skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, 
                                df_2_balanced, df_3_balanced, 
                                df_4_balanced, df_5_balanced, df_6_balanced])
  skin_df_balanced['age'] = skin_df_balanced['age'].fillna(int(skin_df_balanced['age'].mean()))
  
  return skin_df_balanced

skin_df_balanced = balance_data()


image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('HAM10000/', '*', '*.jpg'))}

skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)


def load_image(path):
    img = Image.open(path).convert('RGB')  # asegúrate de que sea RGB
    img = img.resize((SIZE, SIZE))
    return np.asarray(img, dtype=np.uint8)

skin_df_balanced['image'] = skin_df_balanced['path'].progress_map(load_image)

# create dataframes
x_meta = skin_df_balanced[['age', 'sex', 'localization']]
x_img = np.stack(skin_df_balanced['image'].values).astype('float32') / 255.0
y = skin_df_balanced[['label']]

# create pipeline
num_cols = ['age']
cat_cols = ['sex', 'localization']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

x_meta = preprocessor.fit_transform(x_meta)
x_meta = x_meta.toarray()

y = to_categorical(y, num_classes=7)

x_img_train, x_img_test, x_meta_train, x_meta_test, y_train, y_test = train_test_split(x_img, x_meta, y, test_size=0.10, random_state=42)
x_img_train, x_img_val, x_meta_train, x_meta_val, y_train, y_val = train_test_split(x_img_train, x_meta_train, y_train, test_size=0.20, random_state=42)

# print(f'train: {x_meta_train.shape[0]}')
# print(f'val: {x_meta_val.shape[0]}')
# print(f'test: {x_meta_test.shape[0]}')
