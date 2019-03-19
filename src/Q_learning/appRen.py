import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

#propriétés du plateau
width = 5
height = 4
states_n = width*height
actions_n = 4

#positionnement des éléments
hole = [1,2]
wall = [3,0]
end = [3,4]
position = [0,0]

#fonction de récompense imédiate associée a la position sur le plateau
r = np.zeros([height,width])
r[end[0],end[1]]=10
r[hole[0],hole[1]]=-10

#fonction de valeur
Q = np.zeros([states_n,actions_n])

#paramètres de l'epérience
gamma = 0.9
probExp = 0.9

def calculCoeffExp(i,N) :
	"""calcul le coefficient d'exploration.
	Varie de manière linéaire sur N expériences"""
	c = (N-i)/N
	return c

def coord2num(c) :
	"""passage des coordonnées [x,y] au numéro de l'état"""
	return width*c[0]+c[1]

def num2coord(x) :
	"""passage du numéro de l'état aux coordonnées"""
	return [x//width, x%width]

#définition des actions
goUp = [-1,0]
goDown = [1,0]
goRight = [0,1] 
goLeft = [0,-1]


def affichage() :
	"""fonction d'affichage du plateau"""
	str = ""
	for i in range(height) :
		for j in range(width) :
			if np.equal([i,j],position).all() :
				str+="x"
			elif np.equal([i,j],wall).all() :
				str+="#"
			elif np.equal([i,j],end).all() :
				str+="@"
			elif np.equal([i,j],hole).all() :
				str+="o"
			else :
				str+="."
		str+="\n"
	str+="\n"
	print(str)

def sommeQ(Q) :
	"""fait la somme des valeurs de Q"""
	res=0
	for l in Q :
		res += sum(l)
	return res

def movable(pos) :
	"""retourne si la position pos est atteignable"""
	if 0<=pos[0] and pos[0]<height and 0<=pos[1] and pos[1]<width and (pos!=wall).any() :
		return True
	return False

def move(dir) :
	""" 
	déplace la position dans la direction dir.
	0 pour haut, 1 pour droite, 2 pour bas, 3 pour gauche
	"""
	global position
	if dir==0 :
		if movable(np.add(position,goUp)) :
			position = np.add(position,goUp)

	elif dir==1 :
		if movable(np.add(position,goRight)) :
			position = np.add(position,goRight)

	elif dir==2 :
		if movable(np.add(position,goDown)) :
			position = np.add(position,goDown)

	elif dir==3 :
		if movable(np.add(position,goLeft)) :
			position = np.add(position,goLeft)

def realMove(dir,param) :
	""" modélise le déplacement réel, avec une probabilité param de ne pas glisser"""
	newdir = dir
	a = random.random()
	if a<((1-param)/2) :
		newdir = (dir-1)%4
	elif a<(1-param) :
		newdir = (dir+1)%4
	move(newdir)

def isEnded() :
	"""indique si le jeu est fini"""
	if np.equal(position,end).all() or np.equal(position,hole).all() :
		return True
	return False

def choixDirection() :
	"""choisi une direction"""
	a = random.random()
	if a<coeffExploration : #cas d'exploration
		return random.randrange(4)
	else : #cas d'exploitation
		return np.argmax(Q[coord2num(position)])




N = 1000 #nombre d'epériences

#listes utiles pour affichage de la courbe de la somme de Q
tmp = [i for i in range(N+1)]
sumQ = [0]

#début de l'entrainement
for i in range(N) :

	#on se replace au début, le temps est nul, et maj du coefficient d'exploration
	position=[0,0]
	t=0
	coeffExploration = calculCoeffExp(i,N)

	while not isEnded() : #on joue jusqu'à la fin
		
		#maj du temps et du coefficient alpha
		t+=1
		alpha = 1/t

		direction = choixDirection() #choix de la direction dans laquelle on va se déplacer

		#calcul du nouvel état a partir de la position actuelle et de la direction choisie
		oldState = coord2num(position)
		realMove(direction,probExp)
		newState = coord2num(position)

		#maj de la fonction Q
		Q[oldState,direction] = (1-alpha)*Q[oldState,direction] + alpha*(r[position[0],position[1]]+gamma*max(Q[newState]))
	
	
	sumQ.append(sommeQ(Q)) #maj de la somme des valeurs de Q
	
#print(Q)


def affSol():
	"""affiche la solution trouvée a partir de Q"""
	#ne marche pas pour l'instant
	global position
	position = [0,0]
	chemin = [position]

	while not isEnded() :
		direction = np.argmax(Q[coord2num(position)])
		move(direction)
		chemin.append(position)

	print(chemin.index([0,0]).__str__())
	str = ""
	for i in range(height) :
		for j in range(width) :
			for p in chemin :
				if np.equal([i,j],p).all() :
					str+=chemin.index([i,j]).__str__()
			if np.equal([i,j],wall).all() :
				str+="#"
			elif np.equal([i,j],end).all() :
				str+="@"
			elif np.equal([i,j],hole).all() :
				str+="o"
			else :
				str+="."
		str+="\n"
	str+="\n"
	print(str)




def parcours() :
	global position
	position = [0,0]

	while not isEnded() :

		affichage()
		direction = np.argmax(Q[coord2num(position)])
		move(direction)

parcours()

#affichage de l'évolution de la somme des valeurs de Q
plt.plot(tmp, sumQ)
plt.show()
plt.figure()