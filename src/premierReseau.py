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
a = 5
b = 2
noise_scale = .1
x_np = np.random.rand(N,1) #array de N nombres aléatoires entre 0 et 1 (les coordonées x des N points)
#print(x_np)
noise = np.random.normal(scale=noise_scale, size=(N,1)) #array de N bruits blancs gaussiens
y_np = np.reshape(a*x_np + b + noise, (-1)) #points du jeu de données bruités (les coordonnées y des N points)

