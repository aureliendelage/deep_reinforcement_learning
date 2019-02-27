import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import pearsonr
import numpy as np
from mlxtend.data import loadlocal_mnist
tf.InteractiveSession()


'''img = mpimg.imread("/home/etudiants/delage15u/PIDR/telecom_2019_deep_rl/classification_images/0/image_0.png")
if img.dtype == np.float32: # Si le résultat n'est pas un tableau d'entiers
    img = (img * 255).astype(np.uint8)
print("img : ",img)
'''


# On charge les données : X contient 60000 images de 784 pixels, et y contient les 60000 resultats attendus
X, y_charge = loadlocal_mnist(
        images_path='train-images.idx3-ubyte', 
        labels_path='train-labels.idx1-ubyte')

"""
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])
print('valeur associée : %s' % (y[0]))"""

N = 10000
taille_entree = X.shape[1]
X_entre = X[:N]
y_att = np.zeros((N,10))
for i in range(N) :
	y_att[i,y_charge[i]] = 1


hidden1 = 16
hidden2 = 16
hidden3 = 16

with tf.name_scope("placeholders"): #données d'entrée
	  x = tf.placeholder(tf.float32, (N,taille_entree))
	  y = tf.placeholder(tf.float32, (N,10))
with tf.name_scope("layer_1"):
  W1 = tf.Variable(tf.random_normal([taille_entree, hidden1]))  ## 
  b1 = tf.Variable(tf.zeros([hidden1])) ## biais pour les hidden neurones cachés
  layer_1 = tf.sigmoid(tf.matmul(x,W1) + b1)
with tf.name_scope("layer_2"):
  W2 = tf.Variable(tf.random_normal([hidden1, hidden2]))  ## 
  b2 = tf.Variable(tf.zeros([hidden2])) ## biais pour les hidden neurones cachés
  layer_2 = tf.sigmoid(tf.matmul(layer_1,W2) + b2)
with tf.name_scope("layer_3"):
  W3 = tf.Variable(tf.random_normal([hidden2, hidden3]))  ## 
  b3 = tf.Variable(tf.zeros([hidden3])) ## biais pour les hidden neurones cachés
  layer_3 = tf.sigmoid(tf.matmul(layer_2,W3) + b3)
with tf.name_scope("out_layer"):
	  ##print("shape of layer_1 : ",layer_1.shape)
	  Wo = tf.Variable(tf.random_normal([hidden3,10])) ##
	  bo = tf.Variable(tf.zeros([10])) ## biais pour les 2 classes de sortie (output).
	  y_pred = tf.sigmoid(tf.matmul(layer_3,Wo) + bo)
with tf.name_scope("loss"): #fonction de perte
	  l = tf.losses.mean_squared_error(y_att, y_pred)
with tf.name_scope("optim"): #fonction d'optimisation
	  train_op = tf.train.AdamOptimizer(.01).minimize(l)


with tf.name_scope("summaries"): #pour sauvegarder l'évolution de la fonction de perte
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/digit', tf.get_default_graph())


n_steps = 2000
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #entraiement du réseau
  print("entrainement en cours")
  for i in range(n_steps):
    feed_dict = {x: X_entre, y: y_att}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("step %d, loss: %f" % (i, loss))
    train_writer.add_summary(summary, i)


  W1f, b1f, W2f, b2f, W3f, b3f, Wof, bof = sess.run([W1, b1, W2, b2, W3, b3, Wo, bo])

  def sigmoid(x) :
  	return 1./(1+np.exp(-x))

  def predire(i) :
  	lay1 = sigmoid(np.dot(X_entre[i],W1f) + b1f)
  	lay2 = sigmoid(np.dot(lay1,W2f) + b2f)
  	lay3 = sigmoid(np.dot(lay2,W3f) + b3f)
  	pred = sigmoid(np.dot(lay3,Wof) + bof)
  	return pred

  print(predire(0))

  def resultat(tab) :
  	vmax=tab[0]
  	imax=0
  	for i in range(1,10) :
  		if vmax<tab[i] :
  			vmax = tab[i]
  			imax = i
  	return imax

  pred = sess.run(y_pred,feed_dict={x: X_entre})
  totaux = np.zeros(10)
  reussi = np.zeros(10)
  for i in range(N) :
  	totaux[y_charge[i]]+=1
  	if y_charge[i] == resultat(pred[i]) :
  		reussi[y_charge[i]]+=1

  for i in range(10) :
  	print( i , "reconnu à ", 100*reussi[i]/totaux[i], "%")




