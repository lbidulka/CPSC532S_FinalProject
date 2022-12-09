import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score
import lovely_tensors as lt
lt.monkey_patch()

from utils.dataset import AudioSet, genres, moods, genre_indices, mood_indices

SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Computes average F1 score for each class over a dataset split
def eval_F1(classifier, dataloader, labels, classification_thresh):
    sum_f1s = torch.zeros((len(labels)))
    for batch_idx, batch in enumerate(tqdm(dataloader)):
            x = batch[0].to(device)
            y = batch[1].to(device)
            pred = classifier(x)
            thresh_pred = (pred > classification_thresh)*1.0
            batch_f1 = f1_score(y.cpu(), thresh_pred.cpu(), average=None, zero_division=0)    # IS THIS CORRECT ZERO_DIV PARAM???
            sum_f1s += batch_f1
    avg_f1s = (sum_f1s / (batch_idx + 1)).numpy()
    return avg_f1s

# Combines dataset split f1 scores into single f1 score weighted by number of samples in each split
def combine_f1s(dataset_info_dir, task, unbal_train_f1s, bal_train_f1s, test_f1s):
    num_unbal_train = np.sum(np.load(dataset_info_dir + task + "_unbal_count.npy"))
    num_bal_train = np.sum(np.load(dataset_info_dir + task + "_bal_count.npy"))
    num_test = np.sum(np.load(dataset_info_dir + task + "_test_count.npy"))
    num_total= num_unbal_train + num_bal_train + num_test
    comb_f1 = (unbal_train_f1s*num_unbal_train + bal_train_f1s*num_bal_train + test_f1s*num_test) / num_total 
    return comb_f1

# Plots and saves F1 bar chart
def plot_f1s(out_dir, task, count_pos, xs, f1s):
    plt.bar(count_pos, f1s)
    plt.xticks(range(len(xs)), xs, size="small")
    plt.tick_params(axis="x", rotation=90)
    plt.title(task + " F1 Scores")
    plt.xlabel(task)
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(out_dir + task + "_f1s.jpg")
    plt.clf()

def main():
    # Load classifiers & dataset
    checkpoint_path = "./CPSC532S_FinalProject/checkpoints/"
    genre_model = torch.load(checkpoint_path + "genre_classifier.pth")
    genre_model.to(device).eval()
    mood_model = torch.load(checkpoint_path + "mood_classifier.pth")
    mood_model.to(device).eval()
    audioset = AudioSet()

    dataset_info_dir = "./CPSC532S_FinalProject/outputs/dataset_info/"
    out_dir = "./CPSC532S_FinalProject/outputs/metrics/"
    classification_thresh = 0.7 # For F1 calculation

    # First we want to check out per-class performance (F1 score) for each task (genre, mood)
    print("\nEvaluating genre unbal train...")
    genre_unbal_train_f1s = eval_F1(genre_model, audioset.genre_unbal_trainloader, genres, classification_thresh)
    print("Evaluating genre bal train...")
    genre_bal_train_f1s = eval_F1(genre_model, audioset.genre_trainloader, genres, classification_thresh)
    print("Evaluating genre test...")
    genre_test_f1s = eval_F1(genre_model, audioset.genre_testloader, genres, classification_thresh)

    print("\nEvaluating mood unbal train...")
    mood_unbal_train_f1s = eval_F1(mood_model, audioset.mood_unbal_trainloader, moods, classification_thresh)
    print("Evaluating mood bal train...")
    mood_bal_train_f1s = eval_F1(mood_model, audioset.mood_trainloader, moods, classification_thresh)
    print("Evaluating mood test...")
    mood_test_f1s = eval_F1(mood_model, audioset.mood_testloader, moods, classification_thresh)

    genre_f1s = combine_f1s(dataset_info_dir, "genre", genre_unbal_train_f1s, genre_bal_train_f1s, genre_test_f1s)
    mood_f1s = combine_f1s(dataset_info_dir, "mood", mood_unbal_train_f1s, mood_bal_train_f1s, mood_test_f1s)

    # Plot results
    genre_count_pos = torch.arange(0, len(genres))
    mood_count_pos = torch.arange(0, len(moods))

    plot_f1s(out_dir, "Genre", genre_count_pos, genres, genre_f1s)
    plot_f1s(out_dir, "Mood", mood_count_pos, moods, mood_f1s)

    np.save(out_dir + "Genre_f1s.npy", genre_f1s)
    np.save(out_dir + "Mood_f1s.npy", mood_f1s)

if __name__ == "__main__":
    main()