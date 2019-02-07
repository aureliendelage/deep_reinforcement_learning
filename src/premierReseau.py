# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:19:19 2019

@author: barloy2u
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.InteractiveSession()

a = tf.zeros((2,4))
print(a.eval())

