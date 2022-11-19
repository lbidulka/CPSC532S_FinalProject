import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

genre_indices = [216, 217, 219, # Pop, Hip Hop, Rock
                 226, 227, 228, # R&B, Soul, Reggae
                 229, 232, 233, # Country, Funk, Folk
                 234, 235, 236, # Middle Eastern, Jazz, Disco
                 237, 239, 248, # Classical, Electronic, Latin
                 251, 252, 253, # Blues, Chilren's, New-age
                 254, 256, 258, # Vocal, Music of Africa, Christian
                 260, 263, 264, # Music of Asia, Ska, Traditional
                 265] # Independent
mood_indices = [276, 277, 278, # Happy, Funny, Sad
                279, 280, 281, # Tender, Exciting, Angry
                282] # Scary

# Dataset
batch_size = 4096

train_bal_path = "./data/audioset_features/audioset_music_only/bal_train_" 
train_unbal_path = "./data/audioset_features/audioset_music_only/unbal_train_" 
eval_path = "./data/audioset_features/audioset_music_only/eval_" 

audioset_genre_train = TensorDataset(torch.Tensor(np.load(train_bal_path + "genre_data.npy")),
                                     torch.Tensor(np.load(train_bal_path + "genre_labels.npy")))
audioset_genre_unbal_train = TensorDataset(torch.Tensor(np.load(train_unbal_path + "genre_data.npy")),
                                     torch.Tensor(np.load(train_unbal_path + "genre_labels.npy")))
audioset_genre_test = TensorDataset(torch.Tensor(np.load(eval_path + "genre_data.npy")),
                                     torch.Tensor(np.load(eval_path + "genre_labels.npy")))

audioset_genre_trainloader = DataLoader(audioset_genre_train, batch_size=batch_size, shuffle=True)
audioset_genre_unbal_trainloader = DataLoader(audioset_genre_unbal_train, batch_size=batch_size, shuffle=True)
audioset_genre_testloader = DataLoader(audioset_genre_test, batch_size=batch_size, shuffle=True) 


# Models
class genre_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, len(genre_indices))
    
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return torch.sigmoid(self.l3(x))

class mood_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, len(mood_indices))
    
    def forward(self, x):
        return torch.sigmoid(self.l1(x))


# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

G_Classifier = genre_classifier()
G_Classifier = G_Classifier.to(device)
optimizer = torch.optim.Adam(G_Classifier.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10
epoch_printfreq = 2 # Print output frequency

train_losses = []
val_losses = []
f1scores = []

# Training
for epoch in range(epochs):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_f1scores = []    
    if epoch % epoch_printfreq == 0:
        print("Starting epoch: ", epoch, "...")
    G_Classifier.train()
    for batch_idx, batch in enumerate(audioset_genre_trainloader):
        x = batch[0]
        y = batch[1]
        x = x.to(device)
        y = y.to(device)

        pred = G_Classifier(x)
        loss = criterion(pred, y)
        epoch_train_losses.append(loss.detach().cpu())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation
    G_Classifier.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(audioset_genre_unbal_trainloader):
            x = batch[0]
            y = batch[1]
            
            x = x.to(device)
            y = y.to(device)

            pred = G_Classifier(x)
            thresh_pred = (pred > 0.70)*1.0
            loss = criterion(pred, y)
            
            f1 = f1_score(y.cpu(), thresh_pred.cpu(), average=None, zero_division=0)    # IS THIS CORRECT ZERO_DIV PARAM???

            epoch_val_losses.append(loss.detach().cpu())
            epoch_val_f1scores.append(f1)

    epoch_mean_train_loss = np.mean(epoch_train_losses)
    train_losses.append(epoch_mean_train_loss)
    epoch_mean_val_loss = np.mean(epoch_val_losses)
    val_losses.append(epoch_mean_val_loss)
    epoch_mean_val_f1scores = np.mean(epoch_val_f1scores)
    f1scores.append(epoch_mean_val_f1scores)

    if epoch % epoch_printfreq == 0:
        print("Epoch", epoch, 
              "mean train loss: ", epoch_mean_train_loss, 
              ", mean val loss: ", epoch_mean_val_loss, 
              ", mean val f1 score: ", epoch_mean_val_f1scores)

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(f1scores, label="Mean Val F1 (0.7 pred threshold)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.suptitle("Genre Classif. Loss")
plt.legend()
plt.show()  

