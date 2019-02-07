# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:19:19 2019

@author: barloy2u
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
tf.InteractiveSession()

"""
a = tf.zeros((2,4))
#print(a.eval())

randMat = tf.random_uniform((2,3), 0, 1)
print(randMat.eval())
"""

#création d'un jeu de données de N points autour de la droite affine y=ax+b
N = 100
vrai_a = 5
vrai_b = 2
noise_scale = .1
x_np = np.random.rand(N,1) #array de N nombres aléatoires entre 0 et 1 (les coordonées x des N points)
#print(x_np)
noise = np.random.normal(scale=noise_scale, size=(N,1)) #array de N bruits blancs gaussiens
y_np = np.reshape(vrai_a*x_np + vrai_b + noise, (-1)) #points du jeu de données bruités (les coordonnées y des N points)

#Génération d'un graphe TensorFlow
with tf.name_scope("placeholders"): #données d'entrée
  x = tf.placeholder(tf.float32, (N, 1))
  y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"): #poids
  W = tf.Variable(tf.random_normal((1, 1)))
  b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"): #y calculés avec les poids du graphe
  y_pred = tf.matmul(x, W) + b
with tf.name_scope("loss"): #fonction de perte
  l = tf.reduce_sum((y - tf.squeeze(y_pred))**2)
with tf.name_scope("optim"): #fonction d'optimisation
	train_op = tf.train.AdamOptimizer(.001).minimize(l)