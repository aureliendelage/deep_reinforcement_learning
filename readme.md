# Apprentissage par renforcement profond 

Projet de Découverte de la Recherche - Telecom Nancy 2019

Encadrants :

- Olivier Buffet (olivier.buffet@inria.fr - 03 54 95 86 15)
- Vincent Thomas (vincent.thomas@loria.fr - 03 54 95 85 08)

Etudiants :

- Nathan Barloy (nathan.barloy@telecomnancy.net)
- Aurélien Delage (aurelien.delage@telecomnancy.net)

# Sujet

Dans le domaine de l'intelligence artificielle, l'apprentissage automatique (Machine Learning) permet à un ordinateur d'apprendre à effectuer une tâche (de classification, de reconnaissance de forme, ...) sur la base d'exemples qui lui sont fournis. Ces dernières années, des progrès importants ont été faits dans ce domaine à travers les techniques d'apprentissage profond (Deep Learning), lesquelles reposent sur des modèles connexionistes (tels que des réseaux neuronaux). Des travaux ont, entre autres, permis des avancées en apprentissage par renforcement (Reinforcement Learning), c'est-à-dire l'apprentissage par essais-erreurs du comportement d'un agent en interaction avec son environnement (le percevant et pouvant agir sur lui).  On peut citer à ce titre des succès dans le cadre de jeux vidéo classiques et du jeu de Go.

Dans le cadre de ce projet, nous souhaitons reproduire certaines expériences d’apprentissage en utilisant une bibliothèque dédiée. Pour cela,

1. on commencera par se familiariser avec les bases de l'apprentissage par renforcement d'une part (sans approximateurs), comme le Q-learning, et de l'apprentissage profond d'autre part ;
2. on regardera ensuite comment utiliser une bibliothèque disponible (par exemple pytorch en python) en cherchant à résoudre un problème simple de classification à partir d’une base d’exemples fournies (reconnaissance de l’écriture) ;
3. enfin, on utilisera cette bibliothèque pour essayer d’effectuer de l’apprentissage par renforcement  profond (Deep RL) sur un jeu à définir sur lequel conduire des expérimentations (par exemple flappy bird).


# Références

Playing Atari With Deep Reinforcement Learning, de V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller. In NIPS Deep Learning Workshop, 2013.
