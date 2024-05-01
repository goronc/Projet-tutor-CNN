import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchvision.transforms import ToTensor
from torch import optim
import matplotlib.pyplot as plt
import multiprocessing
from traitement_donnee import SpectreDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Données entrainement, validation et test
train_data = DataLoader(SpectreDataset("Donnee50")
, batch_size=4, shuffle=True, num_workers = 4, drop_last = True),
validation_data = DataLoader(SpectreDataset("Donnee30")
, batch_size=4, shuffle=False, num_workers = 4, drop_last = True),
test_data = DataLoader(SpectreDataset("Donnee20")
, batch_size=4, shuffle=False, num_workers = 4, drop_last = True),


def calculate_metrics(predictions, labels, nb_classes):

    conf_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(labels, predictions):
        conf_matrix[t, p] += 1


    TP = conf_matrix.diag()
    TN = torch.zeros(nb_classes)
    FP = torch.zeros(nb_classes)
    FN = torch.zeros(nb_classes)
    for c in range(nb_classes):
        idx = torch.ones(nb_classes, dtype=torch.bool)
        idx[c] = 0
        # all non-class samples classified as non-class
        TN[c] = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP[c] = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN[c] = conf_matrix[c, idx].sum()

    P = TP + FN
    N = TN + FP

    accuracy = torch.sum((TP)) / len(labels)
    sensitivity = TP / P
    specificity = TN / N
        
    return accuracy, sensitivity, specificity


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(5964448, 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32,2)

    def forward(self, x):
        x = self.conv1(x)
        x= F.relu(x)
        x = self.conv2(x)
        x = x.view(4,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, loader, criterion, optimizer):
    model.train()
    for spectres, labels in loader:
        spectres = spectres.to(device)
        labels = labels.to(device)
        out = model(spectres.unsqueeze(1))
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
            out = model(spectres.unsqueeze(1))
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


def main():
    model = CNN()
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1000

    # Training and validation
    for i in range(num_epochs):
        for batch in train_data:
            train(model, batch, loss_function, optimizer)
        for batch in validation_data:
            accuracy, sensitivity, specificity, loss = test(model, batch, loss_function, 2)
        print(f'Epoch: {i}, Accuracy: {accuracy.item():.4f}, Sensibilité: {sensitivity.mean().item():.4f}, Spécificité: {specificity.mean().item():.4f}, Loss: {loss:.4f}')

    # Test
    for batch in test_data:
        accuracy, sensitivity, specificity, loss = test(model, batch, loss_function, 2)
        print(f'Valeur du test: Accuracy: {accuracy.item():.4f}, Sensibilité: {sensitivity.mean().item():.4f}, Spécificité: {specificity.mean().item():.4f}, Loss: {loss:.4f}')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

# fig, ax = plt.subplots()
# ax.plot(epochs, tot_train_loss, label = "Train loss")
# ax.plot(epochs, tot_val_acc, label = "Validation accuracy")
# ax.plot(epochs, tot_val_loss ,label ="Validation loss")
# ax.legend()
# plt.show()