import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

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

train_bal_path = "./data/audioset_features/audioset_music_only/bal_train_" 

genre_tensor_x = torch.Tensor(np.load(train_bal_path + "genre_data.npy"))
genre_tensor_y = torch.Tensor(np.load(train_bal_path + "genre_labels.npy"))

batch_size = 1

audioset_genre_train = TensorDataset(genre_tensor_x, genre_tensor_y) # create your datset
audioset_genre_trainloader = DataLoader(audioset_genre_train, batch_size=batch_size) # create your dataloader

# Models
class genre_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 256)
        self.l2 = nn.Linear(256, len(genre_indices))
    
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        return torch.sigmoid(self.l2(x))

class mood_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, len(mood_indices))
    
    def forward(self, x):
        return torch.sigmoid(self.l1(x))

# Setup
G_Classifier = genre_classifier()
optimizer = torch.optim.Adam(G_Classifier.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

epochs = 5

losses = []

# Training
for epoch in range(epochs):
    epoch_losses = []
    print("Starting epoch: ", epoch)
    for batch_idx, batch in enumerate(audioset_genre_trainloader):
        x = batch[0]
        y = batch[1]

        pred = G_Classifier(x)

        loss = criterion(pred, y)
        epoch_losses.append(loss.detach())

        if batch_idx % 5000 == 0:
            print("Loss in epoch ", epoch, " :", loss.detach())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_mean_loss = np.mean(epoch_losses)
    losses.append(epoch_mean_loss)
    print("Epoch", epoch, "mean loss: ", epoch_mean_loss)

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.suptitle("Genre Classif. Loss")
plt.show()  

