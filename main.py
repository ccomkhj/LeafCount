 
__author__ = "Ankit Patnala"

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import horovod.tensorflow.keras as hvd
from unarypot import *

import csv 
import os

def create_inputs_and_outputs(folder_path,file_name):
    image_folder_dir = os.path.join(folder_path,'train_images')
    folder_dir = os.path.join(os.curdir,folder_path)
    csv_file_path = os.path.join(folder_dir,file_name)
    df = pd.read_csv(csv_file_path,names=['file_name','number_of_leaves'])
    number_of_images = len(df['file_name'])
    image_array = np.empty(shape=(number_of_images,512,512,3))
    label_array = np.empty(shape=(number_of_images,1))
    for idx in range(number_of_images):
        file_name = df['file_name'][idx]
        file_path = os.path.join(image_folder_dir,file_name)
        print(file_path)
        original_image = cv2.imread(file_path,1)
        image_array[idx] = original_image/255.0
        label_array[idx] = df['number_of_leaves'][idx]
    return image_array,label_array
            

class Leaf_counting_model(tf.keras.Model):
    def __init__(self):
        super(Leaf_counting_model,self).__init__()
        self.model = tf.keras.Sequential(
                [tf.keras.layers.Conv2D(32,(3,3),input_shape=(512,512,3)),
                 tf.keras.layers.MaxPooling2D((2,2)),
                 tf.keras.layers.Flatten(),
                 tf.keras.layers.Dense(64,activation='relu'),
                 tf.keras.layers.Dense(1)])
        
        self.model_dense_net =tf.keras.models.load_model('DenseNet')
        #self.model_dense_net.build(input_shape=(512,512,3))
        
        model2 = tf.keras.Sequential()
        model2.add(tf.keras.layers.Flatten())
        model2.add(tf.keras.layers.Dense(64,activation='relu'))
        model2.add(tf.keras.layers.Dense(32,activation='relu'))
        model2.add(tf.keras.layers.Dense(1,activation='relu'))

        inputs = tf.keras.Input(shape=(512,512,3))
        x = self.model_dense_net(inputs)
        for layer in self.model_dense_net.layers[:]:
            layer.trainable = True

        outputs = model2(x)
        self.model3 = tf.keras.Model(inputs,outputs)
        
        print("this is the model with fine tuning")
        
        print(self.model3.summary())
        
        
    def call(self,x):
        #x = self.model_dense_net(x)
        self.model3(x)
        return self.model3(x)

    def compile_and_fit(self,inputs,outputs,batch_size):
        train_size = int(0.9*len(inputs))
        val_size = int(0.1*len(inputs))
        self.model3.compile(optimizer='adam',loss= tf.keras.losses.MAE,metrics=['accuracy'])
        full_dataset = tf.data.Dataset.from_tensor_slices((inputs,outputs))
        full_dataset = full_dataset.shuffle(100).batch(batch_size)
        train_dataset = full_dataset.take(train_size)
        print(len(list(train_dataset)))
        val_dataset = full_dataset.take(val_size)
        print(len(list(val_dataset)))
        early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,mode='min')
        history = self.model3.fit(train_dataset,
                epochs=100,
                validation_data=val_dataset,
                callbacks=[early_stopping])
        print(history.history)
        self.model3.save('Model_with_fine_tuning')

    def output_to_csv(self,x):
        data = self.model3(x)
        with open('valid.csv','w') as write:
            csv_writer = csv.writer(write)
            for val in data:
                csv_writer.writerow(val)

        print(data)

def Horovod_leaf_count_model(model,inputs,outputs,batch_size,epochs=23,learning_rate=0.00001):
        hvd.init()
        print(hvd.rank(),hvd.size())
        train_dataset = tf.data.Dataset.from_tensor_slices((inputs,outputs)).take(epochs*10)
        train_dataset = train_dataset.shuffle(100).batch(batch_size).shard(hvd.size(),hvd.rank())
        model = model
        opt = hvd.DistributedOptimizer(tf.optimizers.Adam(learning_rate))
        model.compile(loss=tf.keras.losses.MeanSquaredError())
        print("i am compiling the model") 
        
        if hvd.rank()==0:
            checkpoint_path = "training_1/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)

            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
        print("i have defined the callbacks")

        callbacks = [
                hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                hvd.callbacks.MetricAverageCallback()]
        if hvd.rank()==0:
            callbacks.append(cp_callback)
        print("Lets fit")

        model.fit(train_dataset,steps_per_epoch=1,
                verbose=2, callbacks=callbacks, epochs = epochs)

        model.save('Model_with_fine_tuning')

if __name__ == "__main__":
    leaf_model = Leaf_counting_model()
    inputs,outputs = create_inputs_and_outputs("cropped_image_new_folder",'train.csv')
    #inputs,outputs = create_inputs_and_outputs_from_praise("../../data/plant_leaf_counting/LCC2020/","train.csv")
    #inputs,outputs = create_inputs_and_outputs_from_praise("set_images","train.csv")
    #Horovod_leaf_count_model(leaf_model,inputs,outputs,16)
    leaf_model.compile_and_fit(inputs,outputs,32)
    leaf_model.output_to_csv(inputs[0:10])

