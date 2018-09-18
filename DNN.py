# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:36:21 2018

@author: admin
"""

import tensorflow
import tflearn

#input data layer
net = tflearn.input_data(shape = [None, 11])

#hidden layers
#3 layers
#every layer have 6 units
net = tflearn.fully_connected(net, 6, activation = 'relu')
net = tflearn.fully_connected(net, 6, activation = 'relu')
net = tflearn.fully_connected(net, 6, activation = 'relu')

#outpu layer
net = tflearn.fully_connected(net, 2, activation = 'softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

X_train[1:100]
model.fit(X_train, y_train, n_epoch = 30, batch_size = 32,
          show_metric = True)

