import os
import random
import shutil
import math

FILELIST = os.listdir("Donnee_padding")

dico = {}
# 1
dico["listR"] = []
# 2
dico["listRp"] = []
# 3
dico["listRm"] = []
# 4
dico["listS"] = []
# 5
dico["listSp"] = []
# 6
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

    nbfichier = longueur * 80 / 100

    random.shuffle(liste)

    for i in range(math.floor(nbfichier)):
        liste_a_deplacer.append(liste[i])

    deplacerFichier(liste_a_deplacer)

    


def deplacerFichier(liste):
    source = os.getcwd() + '/Donnee_padding/'

    destination = os.getcwd() + '/Donnée20/'

    if not os.path.exists(destination):
        os.makedirs(destination)

    for i in range(len(liste)):
        data_list = []
        filepath = os.path.join(source, liste[i])
        try:
            with open(filepath, 'r') as fichier:
                for ligne in fichier:
                    valeurs = ligne.split()
                    data_list.append(int(valeurs[1]))

        except FileNotFoundError:
            print("Le fichier spécifié n'a pas été trouvé.")
            return None

        except Exception as e:
            print("Une erreur s'est produite :", e)
            return None

        with open(destination+liste[i], "w") as fichier_sortie:
            for value in data_list:
                fichier_sortie.write(f"{value}\n")
    


separerFichier(dico)

for i in dico.keys():
    separer20(dico[i])