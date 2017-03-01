# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:37:44 2017

@author: yangshiyu89
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Define dl net
def Neural_Net(train_features, train_targets, val_features, val_targets, test_features):
    
    features = tf.placeholder(tf.float32, shape=[None, train_features.shape[1]])
    targets = tf.placeholder(tf.float32, shape=[None, train_targets.shape[1]])
    
    W_1 = tf.Variable(tf.truncated_normal(shape=[train_features.shape[1], 25], dtype=tf.float32, stddev=0.001))
    b_1 = tf.Variable(tf.zeros(shape=[25], dtype=tf.float32))
    
    W_2 = tf.Variable(tf.truncated_normal(shape=[25, train_targets.shape[1]], dtype=tf.float32, stddev=0.001))
    b_2 = tf.Variable(tf.zeros(shape=[train_targets.shape[1]], dtype=tf.float32))
    
    layer = tf.add(tf.matmul(features, W_1), b_1)
    layer = tf.nn.relu(layer)
    
    predict = tf.add(tf.matmul(layer, W_2), b_2)
    
    loss = tf.reduce_mean(tf.pow(targets - predict, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1001):
            sess.run(optimizer, feed_dict={features:train_features, targets:train_targets})
            if epoch%10 == 0:
                cost_test = sess.run(loss, feed_dict={features:train_features, targets:train_targets})
                cost_val = sess.run(loss, feed_dict={features:val_features, targets:val_targets})
                print("epoch {:4d}; cost_test: {:.4f}; cost_val: {:.4f}".format(epoch, cost_test, cost_val))
        predict_targets = sess.run(predict, feed_dict={features:test_features})
    return predict_targets
if __name__ == "__main__":
    # Load and prepare the data
    data_path = 'Bike-Sharing-Dataset/hour.csv'
    rides = pd.read_csv(data_path)
    
    # Dummy variables
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)
    
    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    
    # Scaling target variables
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std
        
    # Splitting the data into training, testing, and validation sets
    # Save the last 21 days 
    test_data = data[-21*24:]
    data = data[:-21*24]
    
    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
    # Hold out the last 60 days of the remaining data as a validation set
    train_features, train_targets = features[:-60*24], targets[:-60*24]
    val_features, val_targets = features[-60*24:], targets[-60*24:]
    
    
    predict_targets = Neural_Net(train_features, train_targets["cnt"][:, np.newaxis], val_features, val_targets["cnt"][:, np.newaxis], test_features)  
    
    
    # Check the prediction
    fig, ax = plt.subplots(figsize=(8,4))
    mean, std = scaled_features['cnt']
    predictions = predict_targets*std + mean
    ax.plot(predictions[:], label='Prediction')
    ax.plot((test_targets['cnt']*std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()
    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)
