import torch
import torch.nn as nn
import torch.nn.functional as F

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