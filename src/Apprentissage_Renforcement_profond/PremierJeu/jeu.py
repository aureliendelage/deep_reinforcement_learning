import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
tf.InteractiveSession()


##(0,0): coin en bas à gauche
## 0 : haut
## 1 : droite
## 2 : bas
## 3 : gauche

def AtCenter(x,y):
	if (0<x<4):
		if (0<y<4):
			return -1
		if (y==0):
			return 2
		if (y==4):
			return 0
	if (x==0):
		return 3
	if (x==4):
		return 1

def Fonction_Valeur(x,y,n,board,marqued):
	if (marqued[x][y]):
		return -10
	else:
		marqued[x][y] = 1
	##board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1)])
	if (x==4 and y==4):
		Afficher_board(board,n)
		exit(1)
	else:
		if (AtCenter(x,y) == -1):##au centre du jeu
			board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
				board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued),
				board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
				board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])

			return max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
				board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued),
				board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
				board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
		if (AtCenter(x,y) == 0):##bordure haute
			if (x==0):##coin gauche
				board[x][y][x][y] = max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
			if (x==4):##coin droit
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
			if (0<x<4):##au centre de la bordure du haut
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued)])
		if (AtCenter(x,y) == 1):##bordure droite
			if (y==4):##au coin en haut à droite
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
			if (y==0):##au coin en bas à droite
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
			if (0<y<4):##au centre de la bordure droite
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
		if (AtCenter(x,y) == 2):##bordure du bas
			if (x==0):##coin en bas à gauche
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
			if (x==4):##coin en bas à droite
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued)])
			if (0<x<4):##au centre de la bordure du bas
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued)])
		if (AtCenter(x,y) == 3):##bordure de gauche
			if (y==4):##au coin en haut à gauche
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
			if (y==0):##au coin en bas à gauche
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued)])
			if (0<y<4):##au centre de la bordure gauche
				board[x][y] = max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])
				return max([board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,board,marqued),
					board[x][y]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,board,marqued)])


Board = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,1]]



def Afficher_board(board,n):
	print(" \n \n étape : ",n,"board : ")
	print(" | ",board[4][0]," | ",board[4][1]," | ",board[4][2]," | ",board[4][3]," | ",board[4][4]," | ")
	print(" | ",board[3][0]," | ",board[3][1]," | ",board[3][2]," | ",board[3][3]," | ",board[3][4]," | ")
	print(" | ",board[2][0]," | ",board[2][1]," | ",board[2][2]," | ",board[2][3]," | ",board[2][4]," | ")
	print(" | ",board[1][0]," | ",board[1][1]," | ",board[1][2]," | ",board[1][3]," | ",board[1][4]," | ")
	print(" | ",board[0][0]," | ",board[0][1]," | ",board[0][2]," | ",board[0][3]," | ",board[0][4]," | ")


'''
Fonction_probabilite = [[[0.0,0.5,0.5,0.0],[0.0,0.5,0.5,0.5],[0.0,0.5,0.5,0.5],[0.0,0.5,0.5,0.5],[0.0,0.5,0.5,0.0]],
					   [[0.5,0.5,0.5,0.0],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.0,0.5,0.5]],
					   [[0.5,0.5,0.5,0.0],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.0,0.5,0.5]],
					   [[0.5,0.5,0.5,0.0],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.0,0.5,0.5]],
					   [[0.5,0.5,0.0,0.0],[0.5,0.5,0.0,0.5],[0.5,0.5,0.0,0.5],[0.5,0.5,0.0,0.5],[0.5,0.0,0.0,0.5]]]

'''
Fonction_probabilite = [[[0.0,round(random.random(),1),round(random.random(),1),0.0],[0.0,round(random.random(),1),round(random.random(),1),round(random.random(),1)],[0.0,round(random.random(),1),round(random.random(),1),round(random.random(),1)],[0.0,round(random.random(),1),round(random.random(),1),round(random.random(),1)],[0.0,round(random.random(),1),round(random.random(),1),0.0]],
					   [[round(random.random(),1),round(random.random(),1),round(random.random(),1),0.0],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),0.0,round(random.random(),1),round(random.random(),1)]],
					   [[round(random.random(),1),round(random.random(),1),round(random.random(),1),0.0],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),0.0,round(random.random(),1),round(random.random(),1)]],
					   [[round(random.random(),1),round(random.random(),1),round(random.random(),1),0.0],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),round(random.random(),1),round(random.random(),1),round(random.random(),1)],[round(random.random(),1),0.0,round(random.random(),1),round(random.random(),1)]],
					   [[round(random.random(),1),round(random.random(),1),0.0,0.0],[round(random.random(),1),round(random.random(),1),0.0,round(random.random(),1)],[round(random.random(),1),round(random.random(),1),0.0,round(random.random(),1)],[round(random.random(),1),round(random.random(),1),0.0,round(random.random(),1)],[round(random.random(),1),0.0,0.0,round(random.random(),1)]]]

print(" \n \nfonction de probabilité:")
print(" | ",Fonction_probabilite[0][0]," | ",Fonction_probabilite[0][1]," | ",Fonction_probabilite[0][2]," | ",Fonction_probabilite[0][3]," | ",Fonction_probabilite[0][4]," | ")
print(" | ",Fonction_probabilite[1][0]," | ",Fonction_probabilite[1][1]," | ",Fonction_probabilite[1][2]," | ",Fonction_probabilite[1][3]," | ",Fonction_probabilite[1][4]," | ")
print(" | ",Fonction_probabilite[2][0]," | ",Fonction_probabilite[2][1]," | ",Fonction_probabilite[2][2]," | ",Fonction_probabilite[2][3]," | ",Fonction_probabilite[2][4]," | ")
print(" | ",Fonction_probabilite[3][0]," | ",Fonction_probabilite[3][1]," | ",Fonction_probabilite[3][2]," | ",Fonction_probabilite[3][3]," | ",Fonction_probabilite[3][4]," | ")
print(" | ",Fonction_probabilite[4][0]," | ",Fonction_probabilite[4][1]," | ",Fonction_probabilite[4][2]," | ",Fonction_probabilite[4][3]," | ",Fonction_probabilite[4][4]," | ")

marqued = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
Fonction_Valeur(0,0,0,Board,marqued)

