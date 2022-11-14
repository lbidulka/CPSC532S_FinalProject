import torch
import tfrecord
import os

train_bal_path = "./data/audioset_features/audioset_v1_embeddings/bal_train/" 
context_description = {"video_id":"byte" ,"labels": "int"}
sequence_description = {"audio_embedding": "byte"}

vid_count = 0
num_segs = []

for filename in os.listdir(train_bal_path):
    fp = os.path.join(train_bal_path, filename)

    if os.path.isfile(fp):
        loader = tfrecord.tfrecord_loader(fp, None,
                                  context_description,
                                  sequence_description=sequence_description)
        

        for context, sequence_feats in loader:
            data = torch.tensor(sequence_feats["audio_embedding"])
            num_segs.append(data.shape[0])
            labels = torch.tensor(context["labels"])
            vid_count+=1

            # print("ID: ", context["video_id"])
            # print("Labels: ", labels)
            # print(data.shape)
            # print(count)
            # print(sequence_feats["audio_embedding"])

avg_segs = sum(num_segs) / len(num_segs)
print("avg segs: ", avg_segs)
print("num vids: ", vid_count)
print("Total samples: ", avg_segs * vid_count)