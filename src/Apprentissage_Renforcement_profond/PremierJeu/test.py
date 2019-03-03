import sys
import os
import random
import numpy as np

sys.setrecursionlimit(1000)
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

def Fonction_Valeur(x,y,n,Board,marqued):
	print("aurelien : ",Fonction_probabilite[3][4][1]);
	##Afficher_Board(Board,n)
	print(Marqued)
	if (Marqued[x][y]):
		print("marque")
		return Board[x][y]
	##Board[x][y] = max([Board[x][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1)])
	if (x==4 and y==4):
		Afficher_Board(Board,n)
		print("je suis à x==4 et y==4")
		return 1
	else:
		Marqued[x][y] = 1
		a = -2
		b = -2
		c = -2
		d = -2
		try:
			if (Marqued[x+1][y] ==0):
				a = Board[x+1][y]*Fonction_probabilite[x][y][1]+Fonction_Valeur(x+1,y,n+1,Board,Marqued)
				print ("a vaut : ",a)
				Marqued[x+1][y] = 1
			else:
				a = Board[x+1][y]
		except:
			print ("fonct: ",Fonction_probabilite[x][y][1],"Board : ",Board[x][y])
			print ("je ne peux pas aller à droite, x : ",x," y : ",y)
		try:
			if (Marqued[x][y+1] ==0):
				b = Board[x][y+1]*Fonction_probabilite[x][y][0]+Fonction_Valeur(x,y+1,n+1,Board,Marqued)
				print ("b vaut : ",b)
				Marqued[x][y+1] = 1
			else:
				b = Board[x][y+1]
		except:
			print("je ne peux pas aller en haut, x : ",x," y : ",y)
		try:
			if (Marqued[x-1][y] ==0):
				c = Board[x-1][y]*Fonction_probabilite[x][y][3]+Fonction_Valeur(x-1,y,n+1,Board,Marqued)
				print ("c vaut : ",c)
				Marqued[x-1][y] = 1
			else:
				c = Board[x-1][y]
		except:
			print("je ne peux pas aller à gauche, x : ",x," y : ",y)
		try:
			if (Marqued[x][y-1] ==0):
				d = Board[x][y-1]*Fonction_probabilite[x][y][2]+Fonction_Valeur(x,y-1,n+1,Board,Marqued)
				print ("d vaut : ",d)
				Marqued[x][y-1] = 1
			else:
				d = Board[x][y-1]
		except:
			print("je ne peux pas aller en bas, x : ",x," y : ",y)


		print("a : ",a,"b : ",b," c :",c," d : ",d)
		print ("max : ",max(a,b,c,d))
		index = [a,b,c,d].index(max(a,b,c,d))
		print("le max est en ",index," au point : ",x," ",y)
		print("je met à jour le board")
		Board[x][y] = max(a,b,c,d)
		Afficher_Board(Board,n)
		return Board[x][y]


Board = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,1]]



def Afficher_Board(Board,n):
	print(" \n \n étape : ",n,"Board : ")
	print(" | ",Board[0][4]," | ",Board[1][4]," | ",Board[2][4]," | ",Board[2][4]," | ",Board[3][4]," | ",Board[4][4])
	print(" | ",Board[0][3]," | ",Board[1][3]," | ",Board[2][3]," | ",Board[2][3]," | ",Board[3][3]," | ",Board[4][3])
	print(" | ",Board[0][2]," | ",Board[1][2]," | ",Board[2][2]," | ",Board[2][2]," | ",Board[3][2]," | ",Board[4][2])
	print(" | ",Board[0][1]," | ",Board[1][1]," | ",Board[2][1]," | ",Board[2][1]," | ",Board[3][1]," | ",Board[4][1])
	print(" | ",Board[0][0]," | ",Board[1][0]," | ",Board[2][0]," | ",Board[2][0]," | ",Board[3][0]," | ",Board[4][0])


'''
Fonction_probabilite = [[[round(random.random(),2),0.5,0.5,round(random.random(),2)],[round(random.random(),2),0.5,0.5,0.5],[0.0,0.5,0.5,0.5],[0.0,0.5,0.5,0.5],[0.0,0.5,0.5,0.0]],
					   [[0.5,0.5,0.5,0.0],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.0,0.5,0.5]],
					   [[0.5,0.5,0.5,0.0],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.0,0.5,0.5]],
					   [[0.5,0.5,0.5,0.0],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.0,0.5,0.5]],
					   [[0.5,0.5,0.0,0.0],[0.5,0.5,0.0,0.5],[0.5,0.5,0.0,0.5],[0.5,0.5,0.0,0.5],[0.5,0.0,0.0,0.5]]]

'''
Fonction_probabilite = [[[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)]],
					   [[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)]],
					   [[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)]],
					   [[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)]],
					   [[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)],[round(random.random(),2),round(random.random(),2),round(random.random(),2),round(random.random(),2)]]]

print(" \n \nfonction de probabilité:")
print(" | ",Fonction_probabilite[0][4]," | ",Fonction_probabilite[1][4]," | ",Fonction_probabilite[2][4]," | ",Fonction_probabilite[3][4]," | ",Fonction_probabilite[4][4]," | ")
print(" | ",Fonction_probabilite[0][3]," | ",Fonction_probabilite[1][3]," | ",Fonction_probabilite[2][3]," | ",Fonction_probabilite[3][3]," | ",Fonction_probabilite[4][3]," | ")
print(" | ",Fonction_probabilite[0][2]," | ",Fonction_probabilite[1][2]," | ",Fonction_probabilite[2][2]," | ",Fonction_probabilite[3][2]," | ",Fonction_probabilite[4][2]," | ")
print(" | ",Fonction_probabilite[0][1]," | ",Fonction_probabilite[1][1]," | ",Fonction_probabilite[2][1]," | ",Fonction_probabilite[3][1]," | ",Fonction_probabilite[4][1]," | ")
print(" | ",Fonction_probabilite[0][0]," | ",Fonction_probabilite[1][0]," | ",Fonction_probabilite[2][0]," | ",Fonction_probabilite[3][0]," | ",Fonction_probabilite[4][0]," | ")

Marqued = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
Fonction_Valeur(0,0,0,Board,Marqued)

