import torch
import tfrecord
import os
import numpy as np
from tqdm import tqdm

train_bal_path = "./data/audioset_features/audioset_v1_embeddings/bal_train/" 
train_unbal_path = "./data/audioset_features/audioset_v1_embeddings/unbal_train/" 
eval_path = "./data/audioset_features/audioset_v1_embeddings/eval/" 
context_description = {"video_id":"byte" ,"labels": "int", "start_time_seconds": "float", "end_time_seconds": "float"}
sequence_description = {"audio_embedding": "byte"}

train_bal_outpath = "./data/audioset_features/audioset_music_only/bal_train_" 
train_unbal_outpath = "./data/audioset_features/audioset_music_only/unbal_train_" 
eval_outpath = "./data/audioset_features/audioset_music_only/eval_" 

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

print("Diving in...")
for in_path, out_path in zip([train_bal_path, eval_path, train_unbal_path], [train_bal_outpath, eval_outpath, train_unbal_outpath]):
# for in_path, out_path in zip([train_bal_path], [train_bal_outpath]):
    print("\nWorking on: ", in_path)
    samples = []
    ids = []
    times = []
    genre_labels = []
    mood_labels = []
    raw_labels = []
    num_mood = 0
    num_genre = 0
    num_comb = 0
    

    for filename in tqdm(os.listdir(in_path)):
        fp = os.path.join(in_path, filename)

        if os.path.isfile(fp):
            loader = tfrecord.tfrecord_loader(fp, None,
                                    context_description,
                                    sequence_description=sequence_description)

            # Look into each file
            for context, sequence_feats in loader:
                data = sequence_feats["audio_embedding"]
                data_tensor = torch.tensor(np.array(data))
                labels = context["labels"]
                vid_IDs = context["video_id"]
                start_t = context["start_time_seconds"]
                end_t = context["end_time_seconds"]
                
                is_mood = False
                is_genre = False    
                # print("labels: ", labels)

                # We want samples with both genre and mood labels
                if set(labels) & set(genre_indices):
                    num_genre += 1
                    is_genre = True

                if set(labels) & set(mood_indices):
                    num_mood += 1
                    is_mood = True

                if is_genre and is_mood:
                    num_comb += 1
                    # print("Labels: ", labels)
                    # Get the genre labels only
                    pruned_genre_labels = []
                    pruned_mood_labels = []
                    for idx in genre_indices:
                        if idx in labels:
                            pruned_genre_labels.append(1)
                        else:
                            pruned_genre_labels.append(0)      
                
                    # Get the mood labels only
                    for idx in mood_indices:
                        if idx in labels:
                            pruned_mood_labels.append(1)
                        else:
                            pruned_mood_labels.append(0)

                    # print(labels)
                    # print(pruned_genre_labels)
                    # print(pruned_mood_labels)
                    # Get the feature samples, and multiply the genre labels accordingly
                    for feat in data:
                        samples.append(feat)
                        raw_labels.append(labels)
                        ids.append(vid_IDs)
                        times.append(np.concatenate((start_t, end_t)))
                        genre_labels.append(pruned_genre_labels)
                        mood_labels.append(pruned_mood_labels)

    print("Num comb samples: ", num_comb)
    print("Num mood: ", num_mood)
    print("Num genre: ", num_genre)
    print("Saving to np arrays...")

    raw_labels_arr = np.array(raw_labels)
    mood_labels_arr = np.array(mood_labels)
    genre_labels_arr = np.array(genre_labels)
    ids_arr = np.array(ids)
    times_arr = np.array(times)
    samples_arr = np.array(samples)
    # print("raw labels: ", raw_labels)

    np.save(out_path + "comb_raw_labels.npy", raw_labels_arr)
    np.save(out_path + "comb_genre_labels.npy", genre_labels_arr)
    np.save(out_path + "comb_mood_labels.npy", mood_labels_arr)
    np.save(out_path + "comb_ids.npy", ids_arr)
    np.save(out_path + "comb_times.npy", times_arr)
    np.save(out_path + "comb_data.npy", samples_arr)

    # print("genre labels: ", genre_labels_arr.shape)
    # print("mood labels: ", mood_labels_arr.shape)

print("Complete!")