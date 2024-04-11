import os
import random
import shutil
import math

FILELIST = os.listdir("donnee")

dico = {}
dico["listR"] = []
dico["listRp"] = []
dico["listRm"] = []
dico["listS"] = []
dico["listSp"] = []
dico["listSm"] = []


def separerFichier(dico):
    numbers = "0123456789"
    

    for filename in FILELIST:
        if(filename[2] in numbers):
            if(filename[0] == "R"):
                dico["listR"].append(filename)
            else:
                dico["listS"].append(filename)
        else:
            if(filename[5] == "+"):
                if(filename[0] == "R"):
                    dico["listRp"].append(filename)
                else:
                    dico["listSp"].append(filename)
            else:
                if(filename[0] == "R"):
                    dico["listRm"].append(filename)
                else:
                    dico["listSm"].append(filename)


def separer20(liste):

    liste_a_deplacer = []

    longueur = len(liste)

    nbfichier = longueur * 20 / 100

    random.shuffle(liste)

    for i in range(math.floor(nbfichier)):
        liste_a_deplacer.append(liste[i])

    deplacerFichier(liste_a_deplacer)

    


def deplacerFichier(liste):
    source = os.getcwd() + '/donnee/'

    destination = os.getcwd() + '/Donnée20/'

    if not os.path.exists(destination):
        os.makedirs(destination)

    for i in range(len(liste)):
        shutil.move(source + liste[i], destination)
    


separerFichier(dico)

for i in dico.keys():
    separer20(dico[i])