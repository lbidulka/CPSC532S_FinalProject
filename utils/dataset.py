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

moods = ["Happy", "Funny", "Sad", "Tender", "Exciting", "Angry", "Scary"]

genres = ["Pop", "Hip Hop", "Rock",
            "R&B", "Soul", "Reggae",
            "Country", "Funk", "Folk",
            "Middle Eastern", "Jazz", "Disco",
            "Classical", "Electronic", "Latin",
            "Blues", "Children's", "New-age",
            "Vocal", "Music of Africa", "Christian",
            "Music of Asia", "Ska", "Traditional",
            "Independent"]

class AudioSet():
    def __init__(self, batch_size = 4096, inference = False) -> None:
        self.batch_size = batch_size

        self.train_bal_path = "./data/audioset_features/audioset_music_only/bal_train_" 
        self.train_unbal_path = "./data/audioset_features/audioset_music_only/unbal_train_" 
        self.eval_path = "./data/audioset_features/audioset_music_only/eval_" 
        
        if inference:
            inf_train = TensorDataset(torch.Tensor(np.load(self.train_bal_path + "comb_data.npy")),
                                        torch.Tensor(np.load(self.train_bal_path + "comb_genre_labels.npy")),
                                        torch.Tensor(np.load(self.train_bal_path + "comb_mood_labels.npy")),
                                        torch.Tensor(np.load(self.train_bal_path + "comb_ids.npy")),
                                        torch.Tensor(np.load(self.train_bal_path + "comb_times.npy")))
            inf_unbal_train = TensorDataset(torch.Tensor(np.load(self.train_unbal_path + "comb_data.npy")), 
                                                torch.Tensor(np.load(self.train_unbal_path + "comb_genre_labels.npy")),
                                                torch.Tensor(np.load(self.train_unbal_path + "comb_mood_labels.npy")),
                                                torch.Tensor(np.load(self.train_unbal_path + "comb_ids.npy")),
                                                torch.Tensor(np.load(self.train_unbal_path + "comb_times.npy")))
            inf_test = TensorDataset(torch.Tensor(np.load(self.eval_path + "comb_data.npy")),
                                        torch.Tensor(np.load(self.eval_path + "comb_genre_labels.npy")),
                                        torch.Tensor(np.load(self.eval_path + "comb_mood_labels.npy")),
                                        torch.Tensor(np.load(self.eval_path + "comb_ids.npy")),
                                        torch.Tensor(np.load(self.eval_path + "comb_times.npy")))
            self.inf_trainloader = DataLoader(inf_train, batch_size=self.batch_size, shuffle=True)
            self.inf_unbal_trainloader = DataLoader(inf_unbal_train, batch_size=self.batch_size, shuffle=True)
            self.inf_testloader = DataLoader(inf_test, batch_size=self.batch_size, shuffle=True)

        else:
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



