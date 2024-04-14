import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class SpectreDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        data_list = self.lire_spectre(filename)
        classe = find_classe(filename)
        return data_list, classe

    def lire_spectre(self, filename):
        data_list = torch.tensor([])  # Créer un tensor vide
        filepath = os.path.join(self.root_dir, filename)
        
        try:
            with open(filepath, 'r') as fichier:
                for ligne in fichier:
                    nouvel_element = torch.tensor([float(ligne)])  # Convertir la ligne en flottant
                    data_list = torch.cat((data_list, nouvel_element), dim=0)  # Concaténer le nouvel élément au tensor existant

        except FileNotFoundError:
            print("Le fichier spécifié n'a pas été trouvé.")
            return None

        except Exception as e:
            print("Une erreur s'est produite :", e)
            return None

        return data_list


    def data_padding(self):
        padding_x = []
        cpt = 0

        # Parcourir tous les fichiers pour collecter les valeurs de x et y
        for filename in self.file_list:
            cpt+=1
            filepath = os.path.join(self.root_dir, filename)
            data_dict = self.lire_spectre(filepath)
            for key in data_dict.keys():
                if key not in padding_x:
                    padding_x.append(key)
            print(f"Fichier {cpt}/{len(self.file_list)} traité avec succès")
            
        print("Step 1: Done")

        # Créer le dossier "Donnee_padding" s'il n'existe pas déjà
        dossier_padding = os.path.join("Donnee_padding")
        if not os.path.exists(dossier_padding):
            os.makedirs(dossier_padding)

        # Parcourir à nouveau tous les fichiers pour ajouter les valeurs manquantes
        cpt = 0
        for filename in self.file_list:
            cpt+=1
            filepath = os.path.join(self.root_dir, filename)
            data_dict = self.lire_spectre(filepath)

            for value in padding_x:
                if value not in data_dict.keys():
                    data_dict[value] = 0

            # Tri des valeurs du dictionnaire
            sorted_dict = dict(sorted(data_dict.items()))

            # Chemin du fichier de sortie dans le dossier "Donnee_padding"
            file_path = os.path.join(dossier_padding, filename)

            # Écrire les données triées avec les valeurs manquantes ajoutées
            with open(file_path, "w") as fichier_sortie:
                for x, y in sorted_dict.items():
                    fichier_sortie.write(f"{x} {y}\n")
                print(f"Fichier {cpt}/{len(self.file_list)} traité avec succès")


numbers = "0123456789"
def find_classe(filename):
    if(filename[2] in numbers):
        if(filename[0] == "R"):
            return 1
        else:
            return 4
    else:
        if(filename[5] == "+"):
            if(filename[0] == "R"):
                return 2
            else:
                return 3
        else:
            if(filename[0] == "R"):
                return 5
            else:
                return 6

from torch_geometric.loader import DataLoader

# Utilisation
# nom_dossier = "Donnée20"
# dataset = SpectreDataset(nom_dossier)
# test = DataLoader(dataset)
# train_features, train_labels = next(iter(test))
