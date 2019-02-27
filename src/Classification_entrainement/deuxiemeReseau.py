# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:19:19 2019

@author: barloy2u
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import pearsonr
tf.InteractiveSession()

"""
a = tf.zeros((2,4))
#print(a.eval())

randMat = tf.random_uniform((2,3), 0, 1)
print(randMat.eval())
"""

def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation)."""
  return pearsonr(y, y_pred)[0]**2

#création d'un jeu de données de N points autour de la droite affine y=ax+b
N = 100
vrai_a = 5
vrai_b = 2
noise_scale = .1


#Génération du jeu de données
N = 100
# Zeros form a Gaussian centered at (-1, -1)
x_zeros = np.random.multivariate_normal(
    mean=np.array((-1, -1)), cov=.1*np.eye(2), size=(N//2,))
y_zeros = np.zeros((N//2,))
# Ones form a Gaussian centered at (1, 1)
x_ones = np.random.multivariate_normal(
    mean=np.array((1, 1)), cov=.1*np.eye(2), size=(N//2,))
y_ones = np.ones((N//2,))


x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Toy Logistic Regression Data")

# Plot Zeros
plt.scatter(x_zeros[:, 0], x_zeros[:, 1], color="blue")
plt.scatter(x_ones[:, 0], x_ones[:, 1], color="red")
plt.show()


#Génération d'un graphe TensorFlow
with tf.name_scope("placeholders"): #données d'entrée
  x = tf.placeholder(tf.float32, (N, 2))
  y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"): #poids
  W = tf.Variable(tf.random_normal((2, 1)))
  b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"): #y calculés avec les poids du graphe
  y_logit = tf.squeeze(tf.matmul(x, W) + b)
  y_one_prob = tf.sigmoid(y_logit)
  y_pred = tf.round(y_one_prob)
with tf.name_scope("loss"): #fonction de perte
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit,labels=y)
  l = tf.reduce_sum(entropy)
with tf.name_scope("optim"): #fonction d'optimisation
	train_op = tf.train.AdamOptimizer(.1).minimize(l)


with tf.name_scope("summaries"): #pour sauvegarder l'évolution de la fonction de perte
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/lr3-train', tf.get_default_graph())

n_steps = 2000 #nombre d'entrainements
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #entraiement du réseau
  for i in range(n_steps):
    feed_dict = {x: x_np, y: y_np}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("step %d, loss: %f" % (i, loss))
    train_writer.add_summary(summary, i)

  # récupération de w et b
  w_final, b_final = sess.run([W, b])
  print("w1 final %f,w2 final %f,b final %f" % (w_final[0],w_final[1],b_final))

  # prédictions	
  y_pred_np = sess.run(y_pred, feed_dict={x: x_np})

plt.clf()
plt.xlabel("Y-true")
plt.ylabel("Y-pred")
plt.scatter(y_np, y_pred_np)
plt.show()



