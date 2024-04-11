import torch

class Donnees(torch.utils.data.Dataset):
  'Caractérise un jeu de données pour PyTorch'
  def init(self, liste_IDs, labels):
        'Initialisation'
        self.labels = labels
        self.liste_IDs = liste_IDs

  def len(self):
       "Représente le nombre total d'exemples du jeu de données"
        return len(self.liste_IDs)

  def getitem(self, indice):
        'Génère un exemple à partir du jeu de données'
        # Sélection de l'exemple
        ID = self.liste_IDs[indice]

        # Chargement des données et obtention du label
        X = torch.load('donnees/' + ID + '.pt')
        y = self.labels[ID]

        return X, y