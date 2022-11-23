import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from models import mood_classifier
from dataset import AudioSet

SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Dataset
audioset = AudioSet()

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Classifier = mood_classifier()
Classifier = Classifier.to(device)
optimizer = torch.optim.Adam(Classifier.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10
epoch_printfreq = 1 # Print output frequency

train_losses = []
val_losses = []
f1scores = []

# Training
for epoch in range(epochs):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_f1scores = []    
    # Training
    Classifier.train()
    for batch_idx, batch in enumerate(audioset.mood_unbal_trainloader):
        x = batch[0]
        y = batch[1]
        x = x.to(device)
        y = y.to(device)

        pred = Classifier(x)
        loss = criterion(pred, y)
        epoch_train_losses.append(loss.detach().cpu())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Validation
    Classifier.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(audioset.mood_trainloader):
            x = batch[0]
            y = batch[1]
            
            x = x.to(device)
            y = y.to(device)

            pred = Classifier(x)
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

# Save the model
torch.save(Classifier, "./CPSC532S_FinalProject/checkpoints/mood_classifier.pth")

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot(f1scores, label="Mean Val F1 (0.7 pred threshold)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.suptitle("Genre Classif. Loss")
plt.legend()
plt.savefig("./CPSC532S_FinalProject/train_curves/mood_training_curves.jpg")
plt.show()  

