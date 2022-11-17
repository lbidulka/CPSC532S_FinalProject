import torch
import tfrecord
import os
import numpy as np

train_bal_path = "./data/audioset_features/audioset_v1_embeddings/bal_train/" 
eval_path = "./data/audioset_features/audioset_v1_embeddings/eval/" 
context_description = {"video_id":"byte" ,"labels": "int"}
sequence_description = {"audio_embedding": "byte"}

train_bal_outpath = "./data/audioset_features/audioset_music_only/bal_train_" 
eval_outpath = "./data/audioset_features/audioset_music_only/eval_" 

genre_indices = [216, 217, 219, # Pop, Hip Hop, Rock
                 226, 228, 227, # R&B, Reggae, Soul 
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

genre_samples = []
genre_samples_labels = []
mood_samples = []
mood_samples_labels = []

print("Diving in...")
for in_path, out_path in zip([train_bal_path, eval_path], [train_bal_outpath, eval_outpath,]):
    print("\nWorking on: ", in_path)
    for filename in os.listdir(in_path):
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

                if set(labels) & set(genre_indices):
                    # Get the genre labels only
                    pruned_labels = []
                    for label in labels:
                        if label in genre_indices:
                            pruned_labels.append(label)

                    # Get the feature samples, and multiply the labels accordingly
                    for feat in data:
                        genre_samples.append(feat)
                        genre_samples_labels.append(pruned_labels)
                
                if set(labels) & set(mood_indices):
                    # Get the genre labels only
                    pruned_labels = []
                    for label in labels:
                        if label in mood_indices:
                            pruned_labels.append(label)

                    # Get the feature samples, and multiply the labels accordingly
                    for feat in data:
                        mood_samples.append(feat)
                        mood_samples_labels.append(pruned_labels)

    print("Num genre samples: ", len(genre_samples_labels))
    print("Num mood samples: ", len(mood_samples_labels))
    print("Saving to np arrays...")

    genre_samples_labels_arr = np.array(genre_samples_labels)
    genre_samples_arr = np.array(genre_samples)
    np.save(out_path + "genre_labels.npy", genre_samples_labels_arr)
    np.save(out_path + "genre_data.npy", genre_samples_arr)

    mood_samples_labels_arr = np.array(mood_samples_labels)
    mood_samples_arr = np.array(mood_samples)
    np.save(out_path + "mood_labels.npy", mood_samples_labels_arr)
    np.save(out_path + "mood_data.npy", mood_samples_arr)

print("Complete!")