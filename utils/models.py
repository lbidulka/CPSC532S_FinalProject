import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import genre_indices, mood_indices

# Models
class genre_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, len(genre_indices))
    
    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return torch.sigmoid(self.l3(x))

class mood_classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, len(mood_indices))
    
    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return torch.sigmoid(self.l3(x))