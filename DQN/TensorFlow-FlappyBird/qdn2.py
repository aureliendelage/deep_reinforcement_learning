import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import pearsonr
import numpy as np
from mlxtend.data import loadlocal_mnist
import sys
import skimage
sys.path.append("game/")
import wrapped_flappy_bird as game
tf.InteractiveSession()

hidden1 = 16
hidden2 = 16
hidden3 = 16

taille_echantillon = 100
state_temp = [0,0,0,0,0,0]
alpha = 0.1 ##to be changed
gamma = 0.99
coeff_explo = 0.9 ##to be changed
def train_network(batch,taille_batch,taille_entree,N):
  with tf.name_scope("placeholders"): #données d'entrée
    x = tf.placeholder(tf.float32, (taille_echantillon,taille_entree))
    y = tf.placeholder(tf.float32, (taille_echantillon,2))
  with tf.name_scope("layer_1"):
    W1 = tf.Variable(tf.random_normal([taille_entree, hidden1]))  ## 
    b1 = tf.Variable(tf.zeros([hidden1])) ## biais pour les hidden neurones cachés
    layer_1 = tf.sigmoid(tf.matmul(x,W1) + b1)
  with tf.name_scope("layer_2"):
    W2 = tf.Variable(tf.random_normal([hidden1, hidden2]))  ## 
    b2 = tf.Variable(tf.zeros([hidden2])) ## biais pour les hidden neurones cachés
    layer_2 = tf.sigmoid(tf.matmul(layer_1,W2) + b2)
  with tf.name_scope("out_layer"):
      ##print("shape of layer_1 : ",layer_1.shape)
    Wo = tf.Variable(tf.random_normal([hidden2,2])) ##
    bo = tf.Variable(tf.zeros([2])) ## biais pour les 2 classes de sortie (output).
    y_pred = tf.matmul(layer_2,Wo) + bo ## pas de sigmoïde ! On veut des Q-valeurs. 
  with tf.name_scope("losses"): #fonction de perte
    l = tf.losses.mean_squared_error((1-alpha)*y, alpha*y_pred)
  with tf.name_scope("optim"): #fonction d'optimisation
    train_op = tf.train.AdamOptimizer(.2).minimize(l)
  with tf.name_scope("summaries"): #pour sauvegarder l'évolution de la fonction de perte
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prochaine_action = [0,1]
    game_state = game.GameState() ## initialize a game
    batch = []
    etape = 0
    image,reward,_,state  = game_state.frame_step([0,1])
    while (True):
      W1f, b1f, W2f, b2f, Wof, bof = sess.run([W1, b1, W2, b2, Wo, bo])
      if (etape%taille_echantillon == 0 and len(batch)>taille_echantillon):## toutes les 100 étapes, on prend un échantillon du batch, et on rétro-propage le gradient.
        etape = 0; ## on réinitialise
        loss_moyenne = 0 
        for k in range(100): ## on entraîne plusieurs fois?
          ##print("reward : ",reward)
          echantillon = random.sample(batch,taille_echantillon)
          W1f, b1f, W2f, b2f, Wof, bof = sess.run([W1, b1, W2, b2,  Wo, bo])## on change les poids, ie on met à jour le réseau secondaire.
          x_dict = [state]## celui d'en ce-moment
          y_dict = [(np.asarray(Q_valeur(W1f, b1f, W2f, b2f, Wof, bof,state)))*gamma + reward]## celui d'en ce-moment
          for i in range(len(echantillon)-1):## il y a déjà l'état courant.
            ##print("i : reward : ",i,reward)
            x_dict.append(echantillon[i][0]);
            y_dict.append(np.asarray(Q_valeur(W1f, b1f, W2f, b2f, Wof, bof,echantillon[i][0]))*gamma+echantillon[i][2]);
          feed_dict = {x: x_dict,y: y_dict}
          _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
          loss_moyenne = loss_moyenne+loss;
        print("loss moyenne: ",loss_moyenne/1000);

      etape = etape +1;
      ##print("je suis ici, étape :",etape);
      ##game_state.init(game_state)
       ## on récupère les poids du réseau
      action = np.zeros([2])
      action[np.argmax(prochaine_action)] = 1
      action[(np.argmax(prochaine_action)+1)%2] = 0 ## little trick, not needed actually..
      ancien_etat = state;
      if (random.random() > coeff_explo): ## exploration
        print("exploration");
        if (random.random()>0.5): ##soit aller en haut, soit ne rien faire, au hasard.
          action = [0,1]
        else:
          action = [1,0]
      image,reward,_,state  = game_state.frame_step(action) ## /!\ On récupère l'état - et le reward - qui suit l'action "action" !
      ##print("reward : ",reward)
      if (len(batch)>=1000):
        batch.pop(0); ## to be changed
      batch.append([ancien_etat,action,reward,state]) ## on sauvegarde l'état.

      print("le réseau veut sauter à : ",prochaine_action[0],"et ne pas sauter à ",prochaine_action[1]);
  
  

def Q_valeur(W1f,b1f,W2f,b2f,Wof,bof,state) : ##le réseau secondaire, pour l'instant. Rq.: il est constant sur tout une série de mini-batch (100 étapes).
    lay1 = sigmoid(np.dot(state,W1f) + b1f)
    lay2 = sigmoid(np.dot(lay1,W2f) + b2f)
    pred = np.dot(lay2,Wof) + bof ## no sigmoid, trying to evaluate the Q-value of a state, not the politicy.
    return pred
def sigmoid(x) :
    return 1./(1+np.exp(-x))
def main():
  print("cc");
  batch = [[]];
  taille_batch = 1000;
  taille_entree = 6;
  N = taille_batch
  train_network(batch,taille_batch,taille_entree,N);


main();