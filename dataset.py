import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

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

class AudioSet():
    def __init__(self) -> None:
        self.batch_size = 4096
        self.train_bal_path = "./data/audioset_features/audioset_music_only/bal_train_" 
        self.train_unbal_path = "./data/audioset_features/audioset_music_only/unbal_train_" 
        self.eval_path = "./data/audioset_features/audioset_music_only/eval_" 
        
        mood_train = TensorDataset(torch.Tensor(np.load(self.train_bal_path + "mood_data.npy")),
                                            torch.Tensor(np.load(self.train_bal_path + "mood_labels.npy")))
        mood_unbal_train = TensorDataset(torch.Tensor(np.load(self.train_unbal_path + "mood_data.npy")),
                                            torch.Tensor(np.load(self.train_unbal_path + "mood_labels.npy")))
        mood_test = TensorDataset(torch.Tensor(np.load(self.eval_path + "mood_data.npy")),
                                            torch.Tensor(np.load(self.eval_path + "mood_labels.npy")))
        self.mood_trainloader = DataLoader(mood_train, batch_size=self.batch_size, shuffle=True)
        self.mood_unbal_trainloader = DataLoader(mood_unbal_train, batch_size=self.batch_size, shuffle=True)
        self.mood_testloader = DataLoader(mood_test, batch_size=self.batch_size, shuffle=True) 

        genre_train = TensorDataset(torch.Tensor(np.load(self.train_bal_path + "genre_data.npy")),
                                            torch.Tensor(np.load(self.train_bal_path + "genre_labels.npy")))
        genre_unbal_train = TensorDataset(torch.Tensor(np.load(self.train_unbal_path + "genre_data.npy")),
                                            torch.Tensor(np.load(self.train_unbal_path + "genre_labels.npy")))
        genre_test = TensorDataset(torch.Tensor(np.load(self.eval_path + "genre_data.npy")),
                                            torch.Tensor(np.load(self.eval_path + "genre_labels.npy")))
        self.genre_trainloader = DataLoader(genre_train, batch_size=self.batch_size, shuffle=True)
        self.genre_unbal_trainloader = DataLoader(genre_unbal_train, batch_size=self.batch_size, shuffle=True)
        self.genre_testloader = DataLoader(genre_test, batch_size=self.batch_size, shuffle=True) 



