#%%
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import MaxPool2D, Conv2D, Flatten, Dropout, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import matplotlib.pyplot as plt
import os
import seaborn as sns
import plotly


metadata = pd.read_csv('Chest_xray_Corona_Metadata.csv')

metadata.head()
train_data = metadata[metadata['Dataset_type']=='TRAIN']
test_data = metadata[metadata['Dataset_type']=='TEST']

print(train_data.isna().sum())
print(test_data.isna().sum())


train_fill = train_data.fillna('unknown')
test_fill = test_data.fillna('unknown')

'''fig,ax = plt.subplots(3,2, figsize=(20,10))
plt.style.use('seaborn')
sns.countplot('Label', data=train_fill,ax=ax[0,0])
sns.countplot('Label_2_Virus_category',data=train_fill,ax=ax[0,1])
sns.countplot('Label_1_Virus_category',data=train_fill,ax=ax[1,0])
sns.countplot('Label', data=test_fill, ax=ax[1,1])
sns.countplot('Label_2_Virus_category',data=test_fill,ax=ax[2,0])
sns.countplot('Label_2_Virus_category',data=test_fill,ax=ax[2,1])
fig.show()


fig,ax =plt.subplots(2,2, figsize=(20,10))
plt.style.use('seaborn')
sns.countplot('Label',data=train_data,ax=ax[0,0])
sns.countplot('Label_2_Virus_category',data=train_data,ax=ax[0,1])
sns.countplot('Label_1_Virus_category',data=train_data,ax=ax[1,0])
fig.show()
'''

TEST_FOLDER = 'C:/Users/Saif Mathur/Desktop/newModel/images/Coronahack-Chest-XRay-Dataset/test'
TRAIN_FOLDER = 'C:/Users/Saif Mathur/Desktop/newModel/images/Coronahack-Chest-XRay-Dataset/train'

from PIL import Image



final_train_data = train_data[(train_data['Label'] == 'Normal') | 
                              ((train_data['Label'] == 'Pnemonia') & (train_data['Label_2_Virus_category'] == 'COVID-19'))]




final_train_data.groupby('Label').size()

final_train_data['target'] = ['negative' if holder == 'Normal' else 'positive' for holder in final_train_data['Label']]

from sklearn.utils import shuffle
final_train_data= shuffle(final_train_data)


final_validation_data = final_train_data.iloc[1000:, :]
final_train_data = final_train_data.iloc[:1000, :]




train_image_generator = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip = True,
    zoom_range=[0.9,1.25],
    brightness_range=[0.5,1.5]
)

test_image_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_image_generator.flow_from_dataframe(
    dataframe=final_train_data,
    directory=TRAIN_FOLDER,
    x_col='X_ray_image_name',
    y_col='target',
    target_size=((224,224)),
    batch_size=8,
    seed=2020,
    shuffle=True,
    class_mode='binary'
)

validation_generator = train_image_generator.flow_from_dataframe(
    dataframe=final_validation_data,
    directory=TRAIN_FOLDER,
    x_col='X_ray_image_name',
    y_col='target',
    target_size=((224,224)),
    batch_size=8,
    seed=2020,
    shuffle=True,
    class_mode='binary'
)

test_generator = test_image_generator.flow_from_dataframe(
    dataframe=test_data,
    directory=TEST_FOLDER,
    x_col='X_ray_image_name',
    target_size=(224,224),   
    shuffle=False,
    batch_size=16,
    class_mode=None
)


model = Sequential()
model.add(Conv2D(64,(3,3), input_shape=(224,224,3),activation='relu'))
model.add(MaxPool2D((3,3)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((3,3)))
model.add(Dropout(0.4))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((3,3)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((3,3)))
model.add(Dropout(0.2))
model.add(Flatten())


#model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])



history = model.fit_generator(train_generator,validation_data=validation_generator,epochs=10)

#%%
from keras.models import save_model,load_model
save_model(model,'newCovidModel.h5')
model.summary()

