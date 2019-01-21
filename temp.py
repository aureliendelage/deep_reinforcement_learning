import sys, os
import pygame
import random
from pygame import *
from random import *

pygame.init()


screen = pygame.display.set_mode((640, 400))
player = pygame.image.load('flap.gif').convert()
background = pygame.image.load('fond2.png').convert()
tuyau_bas = pygame.image.load('tuy_bas.png').convert()
tuyau_haut = pygame.image.load('tuy_haut.png').convert()
game_over = pygame.image.load('game_over.jpg').convert()

ax=0
ay=-9

swap = 0

count = 0
class GameObject:
		speed = 0
		image = ""
		pos = [0,0]
		def init(self,image,height,speed):
			self.speed = speed
			self.image = image
			self.pos = image.get_rect().move(0, height)
		def move(self):
			self.pos = self.pos.move(self.speed, 0)
			if self.pos[0] > 640:
				self.pos[0] = 0
				return uniform(100,600)
			if self.pos[1] < -400:
				self.pos[1] = 0
			return -1
		def go_up(self,speed_y):
			self.pos = self.pos.move(0,speed_y)

##pygame.display.flip()

objects = []
o = GameObject()
GameObject.init(o,player,200,20)

for x in range(10):

	GameObject.move(o)
	objects.append(o)

BLUE = (0x00, 0xFF, 0x00)
a = uniform(100,600)
screen.blit(tuyau_bas,(a,300))
screen.blit(tuyau_haut,(a,0))
a = GameObject.move(o)

swap = 0
while 1:
	b = GameObject.move(o)
	if b != a and b != -1:
		count += 1
		c = uniform(100,300)
		swap = 1
		##myfont = pygame.font.SysFont("monospace", 16)
		##score_display = myfont.render(count, 1,'))
		##screen.blit(score_display, (100, 100))
		
		a = b
	else:
		swap = 0
	if o.pos[0]+47>a and o.pos[0]<a+79:
		##break
		print("in domaine of collision, with pos : ", o.pos[1])
		if o.pos[1]<150 or o.pos[1]+44>250:
			break
	for event in pygame.event.get():
		if event.type==QUIT:
			sys.exit()
		if event.type == pygame.KEYDOWN:
            # si c'est la touche "haut"
			if event.key == pygame.K_UP:
				GameObject.go_up(o,ay)
			if event.key == pygame.K_DOWN:
				GameObject.go_up(o,-ay)
	if swap == 1:
		font = pygame.font.Font(None, 36)
		text = font.render(str(count), 1, Color("white"))
		textpos = text.get_rect(centerx=background.get_width()/2)
		screen.blit(background,(0,0))
		screen.blit(text, textpos)
		screen.blit(tuyau_bas,(a,250))
		screen.blit(tuyau_haut,(a,0))
		screen.blit(o.image, (o.pos[0], o.pos[1]))
	else:
		font = pygame.font.Font(None, 36)
		text = font.render(str(count), 1, Color("white"))
		textpos = text.get_rect(centerx=background.get_width()/2)
		screen.blit(background,(0,0))
		screen.blit(text, textpos)
		screen.blit(tuyau_bas,(a,250))
		screen.blit(tuyau_haut,(a,0))
		screen.blit(o.image, (o.pos[0], o.pos[1]))
	#pygame.draw.line(screen,Color("white"),(jeu.x,jeu.x+10),(jeu.y,jeu.y-10),1)
	pygame.display.update()
	pygame.time.delay(100)

screen.blit(background,(0,0))
screen.blit(game_over,(100,0))
font = pygame.font.Font(None, 36)
text = font.render("your score is : " + str(count), 1, Color("white"))
textpos = text.get_rect(centerx=background.get_width()/2)
screen.blit(text,textpos)
pygame.display.update()
pygame.time.delay(2000)

	
	
	
	
	
	
	