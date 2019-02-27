import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
tf.InteractiveSession()

nb_elem = 100
N = nb_elem*nb_elem #taille du jeu de données

#x_xor = [[0,0],[0,1],[1,0],[1,1]] ## on déclare les variables d'entrées (placeholders). Stockée en mode [[x11,x12],...] et on fera x11 xor x12
#x_xor = [[0+abs(random.gauss(0,0.25)),0+abs(random.gauss(0,0.25))],[0+abs(random.gauss(0,0.25)),1-abs(random.gauss(0,0.25))],[1-abs(random.gauss(0,0.25)),0+abs(random.gauss(0,0.25))],[1-abs(random.gauss(0,0.25)),1-abs(random.gauss(0,0.25))]] ## on déclare les variables d'entrées (placeholders). Stockée en mode [[x11,x12],...] et on fera x11 xor x12
#x_xor = [[random.uniform(0,0.5),random.uniform(0,0.5)],[random.uniform(0,0.5),random.uniform(0.5,1)],[random.uniform(0.5,1),random.uniform(0,0.5)],[random.uniform(0.5,1),random.uniform(0.5,1)]] ## on déclare les variables d'entrées (placeholders). Stockée en mode [[x11,x12],...] et on fera x11 xor x12
x_xor = np.random.rand(N,2)
y_xor = []
#y_xor = [[0],[1],[1],[0]] ## et leurs résultats par la fonction xor
for p in x_xor :
  if (p[0]<0.5 and p[1]<0.5) or (p[0]>=0.5 and p[1]>=0.5) :
    y_xor.append([0])
  else :
    y_xor.append([1]);
X1 = []
X2 = []

for i in range(N):
    X1.append(x_xor[i][0])
    X2.append(x_xor[i][1])
plt.scatter(X1,X2)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-0.5,1.5)


## affichage des points dans R^2

##plt.show()

hidden = 5

"""hidden = 2
hidden_2 = 2
hidden_3 = 2"""


def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation)."""
  return pearsonr(y, y_pred)[0]**2

#W = tf.Variable(tf.random_normal((2, 1)))



with tf.name_scope("placeholders"): #données d'entrée
	  x = tf.placeholder(tf.float32, (N,2))
	  y = tf.placeholder(tf.float32, (N,1))
with tf.name_scope("layer_1"):
  print("shape of x : ",x.shape)
  W = tf.Variable(tf.random_normal([2, hidden]))  ## 
  b = tf.Variable(tf.zeros([hidden])) ## biais pour les hidden neurones cachés
  layer_1 = tf.sigmoid(tf.matmul(x,W) + b)
with tf.name_scope("out_layer"):
	  ##print("shape of layer_1 : ",layer_1.shape)
	  W1 = tf.Variable(tf.random_normal([hidden_3,1])) ##
	  b1 = tf.Variable(tf.zeros([1])) ## biais pour les 2 classes de sortie (output).
	  yy = tf.matmul(layer_3,W1) + b1
	  y_sigm = tf.sigmoid(yy)
	  y_pred = tf.round(y_sigm)
	  print("shape of y_pred : ",y_pred.shape)
with tf.name_scope("loss"): #fonction de perte
	  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = yy,labels  = y)
	  l = tf.reduce_sum(entropy) ## on minimise l'erreur quadratique moyenne
with tf.name_scope("optim"): #fonction d'optimisation
	  train_op = tf.train.AdamOptimizer(.1).minimize(l)


with tf.name_scope("summaries"): #pour sauvegarder l'évolution de la fonction de perte
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/xor-train', tf.get_default_graph())



n_steps = 5000 #nombre d'entrainements


with tf.Session() as sess:

  reussi = False
  while not reussi :
    sess.run(tf.global_variables_initializer())
    #entraiement du réseau
    for i in range(n_steps):
      feed_dict = {x: x_xor, y: y_xor}
      _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
      print("step %d, loss: %f" % (i, loss))
      train_writer.add_summary(summary, i)
    if loss<500 :
      reussi = True


  
  x_plan = np.linspace(0,1,nb_elem)
  X,Y = np.meshgrid(x_plan,x_plan)
  Z = np.zeros(X.shape)

  x_test = []
  for e1 in x_plan :
    for e2 in x_plan :
      x_test.append([e1,e2])

  res = sess.run(y_pred,feed_dict = {x:x_test})

  for e1 in range(nb_elem) :
    for e2 in range(nb_elem) :
      Z[e1,e2] = res[nb_elem*e1+e2,0]

  plt.pcolor(X,Y,Z)
  plt.show()

  """print("b1:  ",b1[0])
  print("b: ",b)
  print("w:  ",w)
  print("w1:  ",w1)"""
  ##print(y_predit)
  #print("y_prédit: ",y_predr)
  ##print("w final %f, b final %f" % (w_final,b_final))

  # prédictions
''' 
plt.clf()
plt.xlabel("Y-true")
plt.ylabel("Y-pred")
plt.scatter(y_xor, y_pred)
plt.show()
'''