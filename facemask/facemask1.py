import numpy as np
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from keras.models import Sequential
from keras.preprocessing import image

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(150,150),
        batch_size=20 ,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test',
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')

model_saved=model.fit(
        training_set,
        epochs=8,
        validation_data=test_set)

test_image=image.load_img(r'./test/with_mask/x1-with-mask.jpg', target_size=(150,150,3))
test_image
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)

from IPython.display import Image
Image(filename='./test/with_mask/x1-with-mask.jpg') 
pred=model.predict(test_image)[0][0]
print(pred)

test_image=image.load_img(r'./test/without_mask/470.jpg', target_size=(150,150,3))
test_image
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)

Image(filename='./test/without_mask/470.jpg') 
pred=model.predict(test_image)[0][0]
print(pred)