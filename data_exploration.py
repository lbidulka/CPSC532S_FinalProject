import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lovely_tensors as lt
lt.monkey_patch()

from utils.dataset import AudioSet, genres, moods, genre_indices, mood_indices

SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    out_dir = "./CPSC532S_FinalProject/outputs/dataset_info/"
    # Dataset
    audioset = AudioSet()

    # First look at samples per genre and samples per mood
    #
    #   in each of bal_train, unbal_train, and test loaders for each of genre and mood
    #   we want to find num samples per class
    #
    genre_count_pos = torch.arange(0, len(genres))
    genre_unbal_count = torch.zeros((len(genres)))
    genre_bal_count = torch.zeros((len(genres)))
    genre_test_count = torch.zeros((len(genres)))
    genre_unbal_multilabel_count = 0
    genre_bal_multilabel_count = 0
    genre_test_multilabel_count = 0

    mood_count_pos = torch.arange(0, len(moods))
    mood_unbal_count = torch.zeros((len(moods)))
    mood_bal_count = torch.zeros((len(moods)))
    mood_test_count = torch.zeros((len(moods)))
    mood_unbal_multilabel_count = 0
    mood_bal_multilabel_count = 0
    mood_test_multilabel_count = 0

    # GENRES -------------------------------------------------------------------------
    print("Counting genre unbalanced train...")
    for batch_idx, batch in enumerate(tqdm(audioset.genre_unbal_trainloader)):
        y = batch[1]
        sum_y = torch.sum(y, dim=0)
        genre_unbal_count += sum_y
        genre_unbal_multilabel_count += torch.count_nonzero(y.sum(axis=1) - 1) 
    print("Num multilabel: ", genre_unbal_multilabel_count)

    plt.bar(genre_count_pos, genre_unbal_count)
    plt.xticks(range(len(genres)), genres, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Genre unbal Class Sample Counts")
    plt.xlabel("Genres")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "genre_unbal_hist.jpg")
    plt.clf()
    np.save(out_dir + "genre_unbal_count.npy", genre_unbal_count.numpy())

    print("Counting genre balanced train...")
    for batch_idx, batch in enumerate(tqdm(audioset.genre_trainloader)):
        y = batch[1]
        sum_y = torch.sum(y, dim=0)
        genre_bal_count += sum_y
        genre_bal_multilabel_count += torch.count_nonzero(y.sum(axis=1) - 1) 
    print("Num multilabel: ", genre_bal_multilabel_count)

    plt.bar(genre_count_pos, genre_bal_count)
    plt.xticks(range(len(genres)), genres, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Genre bal Class Sample Counts")
    plt.xlabel("Genres")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "genre_bal_hist.jpg")
    plt.clf()
    np.save(out_dir + "genre_bal_count.npy", genre_bal_count.numpy())

    print("Counting genre test...")
    for batch_idx, batch in enumerate(tqdm(audioset.genre_testloader)):
        y = batch[1]
        sum_y = torch.sum(y, dim=0)
        genre_test_count += sum_y
        genre_test_multilabel_count += torch.count_nonzero(y.sum(axis=1) - 1) 
    print("Num multilabel: ", genre_test_multilabel_count)

    plt.bar(genre_count_pos, genre_test_count)
    plt.xticks(range(len(genres)), genres, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Genre test Class Sample Counts")
    plt.xlabel("Genres")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "genre_test_hist.jpg")
    plt.clf()
    np.save(out_dir + "genre_test_count.npy", genre_test_count.numpy())

    mood_total_count = genre_test_count + genre_bal_count + genre_unbal_count
    print("Num Genre Samples: ", mood_total_count.sum().item())
    plt.bar(genre_count_pos, mood_total_count)
    plt.xticks(range(len(genres)), genres, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Total Genre Sample Counts")
    plt.xlabel("Genres")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "total_genre_hist.jpg")
    plt.clf()
    

    # MOOD -----------------------------------------------------------------------
    print("Counting mood unbalanced train...")
    for batch_idx, batch in enumerate(tqdm(audioset.mood_unbal_trainloader)):
        y = batch[1]
        sum_y = torch.sum(y, dim=0)
        mood_unbal_count += sum_y
        mood_unbal_multilabel_count += torch.count_nonzero(y.sum(axis=1) - 1) 
    print("Num multilabel: ", mood_unbal_multilabel_count)

    plt.bar(mood_count_pos, mood_unbal_count)
    plt.xticks(range(len(moods)), moods, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Mood unbal Class Sample Counts")
    plt.xlabel("Moods")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "mood_unbal_hist.jpg")
    plt.clf()
    np.save(out_dir + "mood_unbal_count.npy", mood_unbal_count.numpy())

    print("Counting mood balanced train...")
    for batch_idx, batch in enumerate(tqdm(audioset.mood_trainloader)):
        y = batch[1]
        sum_y = torch.sum(y, dim=0)
        mood_bal_count += sum_y
        mood_bal_multilabel_count += torch.count_nonzero(y.sum(axis=1) - 1) 
    print("Num multilabel: ", mood_bal_multilabel_count)

    plt.bar(mood_count_pos, mood_bal_count)
    plt.xticks(range(len(moods)), moods, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Mood bal Class Sample Counts")
    plt.xlabel("Moods")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "mood_bal_hist.jpg")
    plt.clf()
    np.save(out_dir + "mood_bal_count.npy", mood_bal_count.numpy())

    print("Counting mood test...")
    for batch_idx, batch in enumerate(tqdm(audioset.mood_testloader)):
        y = batch[1]
        sum_y = torch.sum(y, dim=0)
        mood_test_count += sum_y
        mood_test_multilabel_count += torch.count_nonzero(y.sum(axis=1) - 1) 
    print("Num multilabel: ", mood_test_multilabel_count)

    plt.bar(mood_count_pos, mood_test_count)
    plt.xticks(range(len(moods)), moods, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Mood test Class Sample Counts")
    plt.xlabel("Moods")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "mood_test_hist.jpg")
    plt.clf()
    np.save(out_dir + "mood_test_count.npy", mood_test_count.numpy())

    mood_total_count = mood_test_count + mood_bal_count + mood_unbal_count
    print("Num Mood Samples: ", mood_total_count.sum().item())
    plt.bar(mood_count_pos, mood_total_count)
    plt.xticks(range(len(moods)), moods, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title("Total Mood Sample Counts")
    plt.xlabel("Moods")
    plt.ylabel("Num Samples")
    plt.tight_layout()
    plt.savefig(out_dir + "total_mood_hist.jpg")
    plt.clf()

if __name__ == "__main__":
    main()