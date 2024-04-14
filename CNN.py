import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchvision.transforms import ToTensor
from torch import optim
import matplotlib.pyplot as plt
from traitement_donnee import SpectreDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = SpectreDataset("Donnée20")

# Données entrainement, validation et test
train_data = DataLoader(data, batch_size=64, shuffle=True),
validation_data = DataLoader(data, batch_size=64, shuffle=False),
test_data = DataLoader(data, batch_size=64, shuffle=False),


def calculate_metrics(predictions, labels, num_classes):
    # Convertir les prédictions en classes prédites
    _, predicted_classes = torch.max(predictions, 1)
    
    # Initialiser les compteurs pour TP, TN, FP et FN pour chaque classe
    TP = torch.zeros(num_classes)
    TN = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)
    
    # Calculer TP, TN, FP et FN pour chaque classe
    for i in range(num_classes):
        TP[i] = torch.sum((predicted_classes == i) & (labels == i)).item()
        TN[i] = torch.sum((predicted_classes != i) & (labels != i)).item()
        FP[i] = torch.sum((predicted_classes == i) & (labels != i)).item()
        FN[i] = torch.sum((predicted_classes != i) & (labels == i)).item()
    
    # Calculer les valeurs nécessaires pour les métriques
    P = TP + FN
    N = TN + FP
    
    # Calculer les métriques
    accuracy = torch.sum(TP + TN) / torch.sum(P + N)
    sensitivity = TP / P
    specificity = TN / N
    
    return accuracy, sensitivity, specificity


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(186389, 128)  
        self.fc2 = nn.Linear(128, 6)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


num_epochs = 100


def train(model, loader, criterion, optimizer):
    model.train()
    for spectres, labels in loader:
        spectres = spectres.to(device)
        labels = labels.to(device)
        out = model(spectres)
        pred = out.argmax(dim=1)
        loss = criterion(out, labels)
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad() 
    return None
      

def test(model, loader, criterion, num_classes):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for spectres, labels in loader:
            spectres = spectres.to(device)
            labels = labels.to(device)
            out = model(spectres)
            _, predicted_classes = torch.max(out, 1)
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            loss = criterion(out, labels)
            total_loss += loss.item()
    
    accuracy, sensitivity, specificity = calculate_metrics(
        torch.tensor(all_predictions), 
        torch.tensor(all_labels), 
        num_classes
    )
    average_loss = total_loss / len(loader)
    
    return accuracy, sensitivity, specificity, average_loss





# Run for 200 epochs

for i in range(num_epochs):
    for batch in train_data:
        train(model, batch, loss_function, optimizer)
    for batch in validation_data:
        accuracy, sensitivity, specificity, loss = test(model, batch, loss_function, 6)
    print(f'Epoch: {i}, Accuracy: {accuracy:.4f}, Sensibilité: {sensitivity:.4f}, Spécificité: {specificity:.4f}, Loss: {loss:.4f}')

# # Test
accuracy, sensitivity, specificity, loss = test(model, test_data, loss_function)
print(f'Valeur du test: Accuracy: {accuracy:.4f}, Sensibilité: {sensitivity:.4f}, Spécificité: {specificity:.4f}, Loss: {loss:.4f}')

# fig, ax = plt.subplots()
# ax.plot(epochs, tot_train_loss, label = "Train loss")
# ax.plot(epochs, tot_val_acc, label = "Validation accuracy")
# ax.plot(epochs, tot_val_loss ,label ="Validation loss")
# ax.legend()
# plt.show()