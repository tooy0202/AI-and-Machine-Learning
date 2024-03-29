# -*- coding: utf-8 -*-
"""Lab 9 - CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bkh1NWVNYPgniNA_QcFV1t6exkUWK_TT

# **Lab 9 - CNN**
### นายอธิศ สุนทโรดม
### 65543206086-2
"""

classes = ['airplane' ,'automobile' ,'bird' , 'cat' ,
           'deer' ,'dog' , 'frog' , 'horse' , 'ship' ,'truck']

import tensorflow as tf
cifar = tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar.load_data()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu,input_shape=(32,32,3))) # จำนวน filter, ขนาด filter
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Flatten()) # แปลงภาพ input 2 มิติเป็น array 1 มิติขนาด 28x28
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu)) # hidden layer 1
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test,y_test)
# conv-pool-conv-pool: test acc=0.63
# conv-conv: test acc = 0.46 (bad)
# conv-conv-pool: test acc=0.61
# (conv-pool)x3: test acc= 0.59
# (conv-pool)x3 + 1 dense(64): test acc=0.63 (7epochs) (10epochs)=0.65

y_pred = model.predict(x_test)
y_pred[1]

x_train.shape

x_test.shape

import matplotlib.pyplot as plt
plt.imshow(x_train[1])
print(classes[y_train[1].item()])

model.summary()