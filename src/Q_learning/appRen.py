import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt


'''#premier plateau

#propriétés du plateau
width = 5
height = 4
states_n = width*height
actions_n = 4

#positionnement des éléments
hole = [[1,2]]
malusHole = [-10]
wall = [[3,0]]
end = [[3,4]]
bonusEnd = [10]
position = [0,0]

'''


#2nd plateau

#propriétés du plateau
width = 5
height = 5
states_n = width*height
actions_n = 4
nb_train = 1000

#positionnement des éléments
hole = [[2,0]]
malusHole = [-10]
wall = [[1,1],[1,2],[1,3],[2,1],[3,1]]
end = [[0,4],[4,0]]
bonusEnd = [10,20]
position = [0,0]


"""
#3e plateau

#propriétés du plateau
width = 7
height = 6
states_n = width*height
actions_n = 4

#positionnement des éléments
hole = [[1,2],[1,5]]
malusHole = [-10,-30]
wall = [[3,0],[4,4],[2,4],[0,4]]
end = [[3,4],[0,6],[5,6]]
bonusEnd = [10, 70,50]
position = [0,0]"""


#fonction de récompense imédiate associée a la position sur le plateau
r = np.zeros([height,width])
for i in range(len(hole)) :
	r[hole[i][0], hole[i][1]] = malusHole[i]
for i in range(len(end)) :
	r[end[i][0], end[i][1]] = bonusEnd[i]

#fonction de valeur
Q = np.zeros([states_n,actions_n])

#paramètres de l'expérience
gamma = 0.95
probNoSlip = 0.9

def calculEpsilon(i,N) :
	"""calcul le coefficient d'exploration.
	Varie de manière linéaire sur N expériences"""
	c = (N-i)/(N)
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

def isWall(position) :
	for w in wall :
		if np.equal(position,w).all() :
			return True
	return False

def isEnd(position) :
	for e in end :
		if np.equal(position,e).all() :
			return True
	return False

def isHole(position) :
	for h in hole :
		if np.equal(position,h).all() :
			return True
	return False

def affichage() :
	"""fonction d'affichage du plateau"""
	str = ""
	for i in range(height) :
		for j in range(width) :
			if np.equal([i,j],position).all() :
				str+="x"
			elif isWall([i,j]) :
				str+="#"
			elif isEnd([i,j]) :
				str+="@"
			elif isHole([i,j]) :
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
	if 0<=pos[0] and pos[0]<height and 0<=pos[1] and pos[1]<width and not isWall(pos) :
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

def isEnded(t=0) :
	"""indique si le jeu est fini"""
	if (isEnd(position) or (t>states_n*10)) : # on s'arrete aussi au bout d'un certain nombre d'actions
		return True
	return False

def explore() :
	return random.randrange(4)

def exploit() :
	return np.argmax(Q[coord2num(position)])


def choixDirection(epsilon) :
	"""choisi une direction"""
	a = random.random()
	if a<epsilon : #cas d'exploration
		return explore()
	else : #cas d'exploitation
		return exploit()


########################### Entrainement #################################

#N = 5000 #nombre d'expériences

#listes utiles pour affichage de la courbe de la somme de Q

tmp = [i for i in range(nb_train+1)]
sumQ = [0]


def entrainement(N) :
	global Q,position
	Q = np.zeros([states_n,actions_n])

	for i in range(N) :

		#on se replace au début, le temps est nul, et maj du coefficient d'exploration
		position=[0,0]
		epsilon =calculEpsilon(i,N)
		t=0
		
		#maj du coefficient alpha
		alpha = 1/(i+1)

		while not isEnded(t) : #on joue jusqu'à la fin

			t+=1

			direction = choixDirection(epsilon) #choix de la direction dans laquelle on va se déplacer

			#calcul du nouvel état a partir de la position actuelle et de la direction choisie
			oldState = coord2num(position)
			realMove(direction,probNoSlip)
			newState = coord2num(position)

			#maj de la fonction Q
			Q[oldState,direction] = (1-alpha)*Q[oldState,direction] + alpha*(r[position[0],position[1]]+gamma*max(Q[newState]))
		
		
		#sumQ.append(sommeQ(Q)) #maj de la somme des valeurs de Q
		

	######################## Fin de l'entrainement ##################################



def indice(liste,elem) :
	for i in range(len(liste)) :
		if np.equal(elem,liste[i]).all() :
			return i
	return -1


def affSol():
	"""affiche la solution trouvée a partir de Q"""
	global position
	position = np.array([0,0])
	chemin = [position]

	while not isEnded() :
		direction = np.argmax(Q[coord2num(position)])
		move(direction)
		chemin.append(position)

	string = ""
	for i in range(height) :
		for j in range(width) :

			ind = indice(chemin,[i,j])
			
			if isWall([i,j]) :
				string+="#"
			elif isEnd([i,j]) :
				string+="@"
			elif isHole([i,j]) :
				string+="o"
			elif ind>=0 :
				string+=str(ind)
			else :
				string+="."
		string+="\n"
	string+="\n"
	print(string)

def trouver_arrivee() :
	for k in range(len(end)) :
		if np.equal(position,end[k]).all() :
			return k
	return -1


def parcours() :
	global position
	position = [0,0]

	while not isEnded() :

		affichage()
		direction = np.argmax(Q[coord2num(position)])
		move(direction)

for inc in range(10) :
	entrainement(nb_train)
	affSol()
	fin = trouver_arrivee()
	print("fin n°",fin)
"""
#affichage de l'évolution de la somme des valeurs de Q
plt.plot(tmp, sumQ)
plt.show()
plt.figure()
"""
#print(Q)