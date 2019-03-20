import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

#propriétés du plateau
width = 10
height = 6
states_n = width*height
actions_n = 4
t=0

#positionnement des éléments
hole = [[1,2],[4,4],[2,8]]
malusHole = [-15,-20,-20]
wall = [[3,0],[5,8]]
end = [[3,4],[5,3]]
bonusEnd = [1,100]
position = [0,0]

#fonction de récompense imédiate associée a la position sur le plateau
r = np.zeros([height,width])
for i in range(len(hole)) :
	r[hole[i][0], hole[i][1]] = malusHole[i]
for i in range(len(end)) :
	r[end[i][0], end[i][1]] = bonusEnd[i]

#fonction de valeur
Q = np.zeros([states_n,actions_n])

#paramètres de l'epérience
gamma = 0.99
probExp = 0.9

def calculCoeffExp(i,N) :
	"""calcul le coefficient d'exploration.
	Varie de manière linéaire sur N expériences"""
	c = (2*N-i)/(2*N)
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

def isEnded() :
	"""indique si le jeu est fini"""
	if isEnd(position) or isHole(position) :
		return True
	return False

def choixDirection() :
	"""choisi une direction"""
	a = random.random()
	if a<coeffExploration : #cas d'exploration
		return random.randrange(4)
	else : #cas d'exploitation
		return np.argmax(Q[coord2num(position)])




N = 10000 #nombre d'epériences

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




def parcours() :
	global position
	position = [0,0]

	while not isEnded() :

		affichage()
		direction = np.argmax(Q[coord2num(position)])
		move(direction)

affSol()

#affichage de l'évolution de la somme des valeurs de Q
plt.plot(tmp, sumQ)
#plt.show()
#plt.figure()
