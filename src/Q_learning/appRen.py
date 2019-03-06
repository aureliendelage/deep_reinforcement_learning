import sys
import os
import random
import numpy as np

width = 5
height = 4
states_n = width*height
actions_n = 4

map = np.zeros([height,width])
hole = [1,2]
wall = [3,0]
end = [3,4]
position = [0,0]
r = np.zeros([height,width])
r[end[0],end[1]]=10
r[hole[0],hole[1]]=-10

Q = np.zeros([states_n,actions_n])
alpha = 0.1
gamma = 0.9
probExp = 1

def coord2num(c) :
	return width*c[0]+c[1]

def num2coord(x) :
	return [x//width, x%width]

goUp = [0,-1]
goDown = [0,1]
goRight = [1,0]
goLeft = [-1,0]


def affichage() :
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

def movable(pos) :
	if 0<=pos[0] and pos[0]<height and 0<=pos[1] and pos[1]<width and (pos!=wall).all() :
		return True
	return False

def move(dir) : # 0 pour haut, 1 pour droite, 2 pour bas, 3 pour gauche
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
	a = random.random()
	if a<((1-param)/2) :
		dir = (dir-1)%4
	elif a<(1-param) :
		dir = (dir+1)%4
	move(dir)

def isEnded() :
	if np.equal(position,end).all() or np.equal(position,hole).all() :
		return True
	return False

N = 1
for i in range(N) :
	position=[0,0]
	while not isEnded() :
		affichage()
		direction = random.randrange(4)
		oldState = coord2num(position)
		realMove(direction,probExp)
		newState = coord2num(position)
		Q[oldState,direction] = (1-alpha)*Q[oldState,direction] + alpha*(r[position[0],position[1]]+gamma*max(Q[newState]))
		print("Q(",oldState,",",direction,") = ",Q[oldState,direction],"\n\n")
	
print(Q)

