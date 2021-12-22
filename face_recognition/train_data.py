from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# image = image.load_img('datasets/train_data/Albert Einstein/2.jpg')
# plt.imshow(image)
# plt.show()
folderDIR_1 = "../face_recognition/Train_data_new"
folderDIR_2 = "../face_recognition/Valid_data_new"

train_datagen = ImageDataGenerator(rescale=1./255, height_shift_range=0.2, width_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, rotation_range=90, shear_range=0.2, validation_split=0.1)
valid_datagen = ImageDataGenerator(rescale=1./255, height_shift_range=0.2, width_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, rotation_range=90, shear_range=0.2, validation_split=0.1)
train_data_generator = train_datagen.flow_from_directory(folderDIR_1, target_size=(224, 224), shuffle=True, batch_size=32, subset='training')
valid_data_generator = valid_datagen.flow_from_directory(folderDIR_2, target_size=(224, 224), shuffle=True, batch_size=32, subset='validation')

print(train_data_generator.class_indices)
input_shape = (224, 224, 3)
epochs = 200
# build models
model = Sequential()
# model.add(Conv2D(16, kernel_size=3,input_shape=input_shape, activation='relu'))
# model.add(Conv2D(16, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model = Sequential()
# model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.8))
# model.add(Dense(10, activation='softmax'))

# pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights=None)
# last_layer = pre_trained_model.get_layer('block5_pool')
# last_output = last_layer.output

# # x = Flatten()(last_output)
# x = GlobalMaxPooling2D()(last_output)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.8)(x)
# x = Dense(10, activation='softmax')(x)

# model = Model(pre_trained_model.input, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print('Start Training')
model.fit(train_data_generator, epochs=epochs, validation_data=valid_data_generator)
model.save("my_model1.h5")

# data without preprocessing
# Epoch 1/20
# 2021-10-17 14:19:29.527396: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 102400000 exceeds 10% of free system memory.
# 2021-10-17 14:19:29.661112: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 102400000 exceeds 10% of free system memory.
# 2021-10-17 14:19:30.207976: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 51200000 exceeds 10% of free system memory.
# 2021-10-17 14:19:30.425357: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 51200000 exceeds 10% of free system memory.
# 2021-10-17 14:19:33.706952: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 46080000 exceeds 10% of free system memory.
# 20/20 [==============================] - 151s 7s/step - loss: 2.3443 - accuracy: 0.1094 - val_loss: 2.3010 - val_accuracy: 0.0923
# Epoch 2/20
# 20/20 [==============================] - 148s 7s/step - loss: 2.3041 - accuracy: 0.0781 - val_loss: 2.3008 - val_accuracy: 0.1077
# Epoch 3/20
# 20/20 [==============================] - 155s 8s/step - loss: 2.2975 - accuracy: 0.1458 - val_loss: 2.3005 - val_accuracy: 0.1077
# Epoch 4/20
# 20/20 [==============================] - 153s 8s/step - loss: 2.3013 - accuracy: 0.1198 - val_loss: 2.2968 - val_accuracy: 0.1769
# Epoch 5/20
# 20/20 [==============================] - 200s 10s/step - loss: 2.2966 - accuracy: 0.1500 - val_loss: 2.2932 - val_accuracy: 0.1846
# Epoch 6/20
# 20/20 [==============================] - 148s 7s/step - loss: 2.3026 - accuracy: 0.1198 - val_loss: 2.2957 - val_accuracy: 0.1846
# Epoch 7/20
# 20/20 [==============================] - 146s 7s/step - loss: 2.2957 - accuracy: 0.1354 - val_loss: 2.2946 - val_accuracy: 0.1846
# Epoch 8/20
# 20/20 [==============================] - 144s 7s/step - loss: 2.2970 - accuracy: 0.0990 - val_loss: 2.2964 - val_accuracy: 0.1077
# Epoch 9/20
# 20/20 [==============================] - 169s 9s/step - loss: 2.3007 - accuracy: 0.1250 - val_loss: 2.2982 - val_accuracy: 0.1077
# Epoch 10/20
# 20/20 [==============================] - 162s 8s/step - loss: 2.3004 - accuracy: 0.1615 - val_loss: 2.2964 - val_accuracy: 0.1077
# Epoch 11/20
# 20/20 [==============================] - 170s 9s/step - loss: 2.2923 - accuracy: 0.1146 - val_loss: 2.2959 - val_accuracy: 0.1769
# Epoch 12/20
# 20/20 [==============================] - 156s 8s/step - loss: 2.2964 - accuracy: 0.0885 - val_loss: 2.2947 - val_accuracy: 0.1846
# Epoch 13/20
# 20/20 [==============================] - 151s 8s/step - loss: 2.2931 - accuracy: 0.1094 - val_loss: 2.2969 - val_accuracy: 0.1769
# Epoch 14/20
# 20/20 [==============================] - 153s 8s/step - loss: 2.3065 - accuracy: 0.0833 - val_loss: 2.2961 - val_accuracy: 0.1846
# Epoch 15/20
# 20/20 [==============================] - 151s 8s/step - loss: 2.2965 - accuracy: 0.1146 - val_loss: 2.2929 - val_accuracy: 0.1846
# Epoch 16/20
# 20/20 [==============================] - 153s 8s/step - loss: 2.3005 - accuracy: 0.1094 - val_loss: 2.2948 - val_accuracy: 0.1769
# Epoch 17/20
# 20/20 [==============================] - 155s 8s/step - loss: 2.3026 - accuracy: 0.0990 - val_loss: 2.2924 - val_accuracy: 0.1077
# Epoch 18/20
# 20/20 [==============================] - 176s 9s/step - loss: 2.2947 - accuracy: 0.1250 - val_loss: 2.2941 - val_accuracy: 0.1077
# Epoch 19/20
# 20/20 [==============================] - 157s 8s/step - loss: 2.2935 - accuracy: 0.1354 - val_loss: 2.2962 - val_accuracy: 0.1077
# Epoch 20/20
# 20/20 [==============================] - 155s 8s/step - loss: 2.2987 - accuracy: 0.1094 - val_loss: 2.2963 - val_accuracy: 0.1077

# data with preprocessing and vgg-face to 256 filters convolution
# Epoch 1/20
# 2021-10-24 12:32:35.154954: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 231211008 exceeds 10% of free system memory.
# 2021-10-24 12:32:35.252653: W tensorflow/core/framework/cpu_allocator_impl.cc:80] Allocation of 231211008 exceeds 10% of free system memory.
# 6/6 [==============================] - 226s 43s/step - loss: 2.7975 - accuracy: 0.0774 - val_loss: 2.3061 - val_accuracy: 0.1045
# Epoch 2/20
# 6/6 [==============================] - 195s 32s/step - loss: 2.3011 - accuracy: 0.1310 - val_loss: 2.3022 - val_accuracy: 0.0821
# Epoch 3/20
# 6/6 [==============================] - 188s 31s/step - loss: 2.3038 - accuracy: 0.1071 - val_loss: 2.3080 - val_accuracy: 0.0970
# Epoch 4/20
# 6/6 [==============================] - 202s 33s/step - loss: 2.2956 - accuracy: 0.1250 - val_loss: 2.3061 - val_accuracy: 0.0970
# Epoch 5/20
# 6/6 [==============================] - 196s 32s/step - loss: 2.2976 - accuracy: 0.1250 - val_loss: 2.3045 - val_accuracy: 0.0970
# Epoch 6/20
# 6/6 [==============================] - 191s 32s/step - loss: 2.3038 - accuracy: 0.0774 - val_loss: 2.3069 - val_accuracy: 0.0896
# Epoch 7/20
# 6/6 [==============================] - 140s 22s/step - loss: 2.3012 - accuracy: 0.1250 - val_loss: 2.3024 - val_accuracy: 0.0896
# Epoch 8/20
# 6/6 [==============================] - 82s 14s/step - loss: 2.2992 - accuracy: 0.1250 - val_loss: 2.3027 - val_accuracy: 0.0896
# Epoch 9/20
# 6/6 [==============================] - 75s 12s/step - loss: 2.2929 - accuracy: 0.1607 - val_loss: 2.3107 - val_accuracy: 0.0896
# Epoch 10/20
# 6/6 [==============================] - 72s 12s/step - loss: 2.3081 - accuracy: 0.1131 - val_loss: 2.3045 - val_accuracy: 0.0896
# Epoch 11/20
# 6/6 [==============================] - 76s 13s/step - loss: 2.2963 - accuracy: 0.1250 - val_loss: 2.3029 - val_accuracy: 0.0896
# Epoch 12/20
# 6/6 [==============================] - 71s 12s/step - loss: 2.2919 - accuracy: 0.1369 - val_loss: 2.3042 - val_accuracy: 0.0896
# Epoch 13/20
# 6/6 [==============================] - 76s 13s/step - loss: 2.3039 - accuracy: 0.1131 - val_loss: 2.3015 - val_accuracy: 0.0896
# Epoch 14/20
# 6/6 [==============================] - 76s 13s/step - loss: 2.2749 - accuracy: 0.1190 - val_loss: 2.2802 - val_accuracy: 0.0896
# Epoch 15/20
# 6/6 [==============================] - 75s 14s/step - loss: 2.2365 - accuracy: 0.1131 - val_loss: 2.2405 - val_accuracy: 0.0896
# Epoch 16/20
# 6/6 [==============================] - 77s 13s/step - loss: 2.2503 - accuracy: 0.1607 - val_loss: 2.2565 - val_accuracy: 0.0896
# Epoch 17/20
# 6/6 [==============================] - 91s 16s/step - loss: 2.2635 - accuracy: 0.1250 - val_loss: 2.2781 - val_accuracy: 0.0896
# Epoch 18/20
# 6/6 [==============================] - 92s 15s/step - loss: 2.2150 - accuracy: 0.1310 - val_loss: 2.2351 - val_accuracy: 0.0896
# Epoch 19/20
# 6/6 [==============================] - 85s 14s/step - loss: 2.2249 - accuracy: 0.1012 - val_loss: 2.2830 - val_accuracy: 0.0896
# Epoch 20/20
# 6/6 [==============================] - 80s 13s/step - loss: 2.2828 - accuracy: 0.1369 - val_loss: 2.3045 - val_accuracy: 0.0896

# lần thứ 3 train với epochs = 100
# Epoch 100/100
# 6/6 [==============================] - 8s 1s/step - loss: 1.2955 - accuracy: 0.5952 - val_loss: 1.5134 - val_accuracy: 0.5075

# lần thứ 4 train với epochs = 100
# Epoch 96/100
# 6/6 [==============================] - 10s 2s/step - loss: 1.1873 - accuracy: 0.6012 - val_loss: 1.4448 - val_accuracy: 0.5299

# lần thứ 5 train với epochs = 100 
# Epoch 96/100
# 5/5 [==============================] - 4s 783ms/step - loss: 1.2849 - accuracy: 0.5742 - val_loss: 1.4988 - val_accuracy: 0.4000

# lần thứ 6 train với model sau đây
# model = Sequential()
# model.add(Conv2D(16, kernel_size=3,input_shape=input_shape, activation='relu'))
# model.add(Conv2D(16, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))
# Epoch 99/100
# 5/5 [==============================] - 8s 2s/step - loss: 1.0317 - accuracy: 0.6581 - val_loss: 1.8943 - val_accuracy: 0.6000

