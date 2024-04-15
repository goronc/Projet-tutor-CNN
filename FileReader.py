import os
import random
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


def separer(liste):

    liste_a_deplacer20 = []
    liste_a_deplacer30 = []
    liste_a_deplacer50 = []

    longueur = len(liste)

    nbfichier20 = math.floor(longueur * 20 / 100)
    nbfichier30 = math.floor(longueur * 30 / 100)

    random.shuffle(liste)

    for i in range(longueur):
        if i < nbfichier20:
            liste_a_deplacer20.append(liste[i])
        elif i < nbfichier20 + nbfichier30:
            liste_a_deplacer30.append(liste[i])
        else:
            liste_a_deplacer50.append(liste[i])

    deplacerFichier(liste_a_deplacer20, '20')
    deplacerFichier(liste_a_deplacer30, '30')
    deplacerFichier(liste_a_deplacer50, '50')

    
def deplacerFichier(liste, pourcentage):
    source = os.getcwd() + '/donnee/'

    destination = os.getcwd() + '/Donnee' + pourcentage + '/'

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
    separer(dico[i])