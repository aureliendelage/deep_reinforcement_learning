import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
tf.InteractiveSession()


x_xor = [[0,0],[0,1],[1,0],[1,1]] ## on déclare les variables d'entrées (placeholders). Stockée en mode [[x11,x12],...] et on fera x11 xor x12
y_xor = [0,1,1,0] ## et leurs résultats par la fonction xor
X1 = []
X2 = []
'''
for i in range(4):
	X1.append(x_xor[i][0])
	X2.append(x_xor[i][1])
plt.scatter(X1,X2)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-0.5,1.5)
'''

def pearson_r2_score(y, y_pred):
  """Computes Pearson R^2 (square of Pearson correlation)."""
  return pearsonr(y, y_pred)[0]**2

W = tf.Variable(tf.random_normal((2, 1)))

## affichage des points dans R^2

##plt.show()


nombre_neurones_cachés = 3




#Génération d'un graphe TensorFlow
with tf.name_scope("placeholders"): #données d'entrée
  x = tf.placeholder(tf.float32, shape = [4,2])
  y = tf.placeholder(tf.float32, shape = [1,2])
with tf.name_scope("layer_1"):
  print("shape of x : ",x.shape)
  W = tf.Variable(tf.random_normal([2, 3]))  ## 1*2 x 2x3 = 1*3
  b = tf.Variable(tf.random_normal([1,3])) ## biais pour les 3 neurones cachés
  layer_1 = tf.matmul(x,W) + b
with tf.name_scope("out_layer"):
  print("shape of layer_1 : ",layer_1.shape)
  W = tf.Variable(tf.random_normal([3,2])) ## 1*3 x 3*2  = 1*2 <=> 2 classes ! ( 0 ou 1 )
  b = tf.Variable(tf.random_normal([1,2])) ## biais pour les 2 classes de sortie (output).
  y = tf.matmul(layer_1,W) + b
  y_sigm = tf.sigmoid(y)
  y_pred = tf.round(y_sigm)
  print("shape of y_pred : ",y_pred.shape)
with tf.name_scope("loss"): #fonction de perte
  l = tf.reduce_sum((y - tf.squeeze(y_pred))**2) ## on minimise l'erreur quadratique moyenne
with tf.name_scope("optim"): #fonction d'optimisation
	train_op = tf.train.AdamOptimizer(.01).minimize(l)## c'est un optimiseur basé sur la descente de gradient. (Rq.: une norme n'est pas différentiable en 0,..)


with tf.name_scope("summaries"): #pour sauvegarder l'évolution de la fonction de perte
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/lr2-train', tf.get_default_graph())


n_steps = 5000 #nombre d'entrainements

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #entraiement du réseau
  for i in range(n_steps):
    feed_dict = {x: x_xor, y: y_xor}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    ##print("step %d, loss: %f" % (i, loss))
    train_writer.add_summary(summary, i)

  # récupération de w et b
  w_final, b_final = sess.run([W, b])
  print("w final %f, b final %f" % (w_final,b_final))

  # prédictions	
  y_pred = sess.run(y_pred, feed_dict={x: x_np})

plt.clf()
plt.xlabel("Y-true")
plt.ylabel("Y-pred")
plt.scatter(y_xor, y_pred)
plt.show()


