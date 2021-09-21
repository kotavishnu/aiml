# Standard Library
import argparse
import random

# Third Party
import numpy as np

# smdebug modification: Import smdebug support for Tensorflow
import smdebug.tensorflow as smd
import tensorflow.compat.v2 as tf

from tensorflow.keras.utils import to_categorical

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense
from keras.models import Sequential
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import boto3
import os
import botocore
from io import BytesIO 
import zipfile

def train(batch_size, epoch, model, hook):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    X_train = train_datagen.flow_from_directory(
            './tmp/train',
            target_size=(150,150),
            batch_size=20 ,
            class_mode='binary')

    Y_train = test_datagen.flow_from_directory(
            './tmp/test',
            target_size=(150,150),
            batch_size=20,
            class_mode='binary')

    # register hook to save the following scalar values
    hook.save_scalar("epoch", epoch)
    hook.save_scalar("batch_size", batch_size)
    hook.save_scalar("train_steps_per_epoch", len(X_train) / batch_size)
    hook.save_scalar("valid_steps_per_epoch", len(Y_train) / batch_size)

    model.fit(
        X_train,
        epochs=epoch,
        validation_data=Y_train,
  
        # smdebug modification: Pass the hook as a Keras callback
        callbacks=[hook],
    )


def main():
    BUCKET_NAME = "xxx"
    ZIP_FILE="facemask/facemask.zip"
    download_zip_s3(BUCKET_NAME,ZIP_FILE,dev_resource)
    print("completed")
  
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
    #model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    # smdebug modification:
    # Create hook from the configuration provided through sagemaker python sdk.
    # This configuration is provided in the form of a JSON file.
    # Default JSON configuration file:
    # {
    #     "LocalPath": <path on device where tensors will be saved>
    # }"
    # Alternatively, you could pass custom debugger configuration (using DebuggerHookConfig)
    # through SageMaker Estimator. For more information, https://github.com/aws/sagemaker-python-sdk/blob/master/doc/amazon_sagemaker_debugger.rst
    hook = smd.KerasHook.create_from_json_file()

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # start the training.
    train(5, 4, model, hook)

dev_resource=boto3.resource('s3')  
def download_zip_s3(bucket,filename,dev_resource):
    zip_obj = dev_resource.Object(bucket_name=bucket, key=filename) 
    print("zip_obj=",zip_obj) 
    buffer = BytesIO(zip_obj.get()["Body"].read()) 
    z = zipfile.ZipFile(buffer) 
    S3_UNZIPPED_FOLDER='./tmp/'

    for filename in z.namelist(): 
        fil=S3_UNZIPPED_FOLDER+"/"+filename
        print("filename is",filename)
        if filename.endswith('/'):
            parent_dir=os.getcwd()
            print("pwd ",parent_dir)
            path = os.path.join(parent_dir, filename)
            os.makedirs(S3_UNZIPPED_FOLDER+filename,exist_ok=False)
            continue
        Body=z.open(filename).read()
        f = open(S3_UNZIPPED_FOLDER+filename, "wb")
        f.write(Body)
        f.close()

    print("Done Unzipping ") 


if __name__ == "__main__":
    main()
