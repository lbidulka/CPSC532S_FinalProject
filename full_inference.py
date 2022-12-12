import torch
import numpy as np
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import pipeline
import lovely_tensors as lt
lt.monkey_patch()
import json

from utils.models import genre_classifier, mood_classifier
from utils.dataset import AudioSet, moods, genres

# SEED = 1234
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

def main():
    # Force a sample of some type for analysis? Leave empy to randomly select
    FORCED_MOOD = ["Funny", "Sad", "Angry", "Scary"] 
    # ["Happy", "Funny", "Sad", "Tender", "Exciting", "Angry", "Scary"]
    FORCED_GENRE = ["Country"]
    #["Pop", "Hip Hop", "Rock",
    # "R&B", "Soul", "Reggae",
    # "Country", "Funk", "Folk",
    # "Middle Eastern", "Jazz", "Disco",
    # "Classical", "Electronic", "Latin",
    # "Blues", "Children's", "New-age",
    # "Vocal", "Music of Africa", "Christian",
    # "Music of Asia", "Ska", "Traditional",
    # "Independent"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load classifiers
    checkpoint_path = "./CPSC532S_FinalProject/checkpoints/"
    mood_model = torch.load(checkpoint_path + "mood_classifier.pth")
    genre_model = torch.load(checkpoint_path + "genre_classifier.pth")

    # Load a sample
    print("Loading dataset...", end=" ")
    audioset = AudioSet(batch_size=1, inference=True)
    print("done!\n")
    print("Searching for specified labels...")
    while 1:
        batch = next(iter(audioset.inf_unbal_trainloader))
        feature, genre, mood, vid_id, times = batch
        feature = feature.to(device)
        mood_idxs = torch.nonzero(mood.squeeze()).reshape(-1,).tolist()
        genre_idxs = torch.nonzero(genre.squeeze()).reshape(-1,).tolist()
        GT_moods = [moods[idx] for idx in mood_idxs]
        GT_genres = [genres[idx] for idx in genre_idxs]
        found_mood = True if ((set(FORCED_MOOD) & set(GT_moods)) or not FORCED_MOOD) else False
        found_genre = True if ((set(FORCED_GENRE) & set(GT_genres)) or not FORCED_GENRE) else False
        if found_mood and found_genre: break

    vid_url = ""
    for id in vid_id[0].tolist():
        vid_url += chr(int(id))
    vid_url = "https://www.youtube.com/watch?v=" + vid_url + "&t=" + str(int(times[0,0].item())) + "s"
    print("Input video URL: ", vid_url)
    print("Time from {}s to {}s\n".format(times[0,0].item(), times[0,1].item()))

    # Get mood/genre of feature
    NUM_MOOD_PREDS = 2
    NUM_GENRE_PREDS = 3
    
    pred_mood = mood_model(feature)
    pred_genre = genre_model(feature)
    _, pred_mood_idxs = torch.topk(pred_mood, NUM_MOOD_PREDS)
    _, pred_genre_idxs = torch.topk(pred_genre, NUM_GENRE_PREDS)
    pred_mood_idxs = pred_mood_idxs.reshape(-1,).tolist()
    pred_genre_idxs = pred_genre_idxs.reshape(-1,).tolist()
    pred_moods = [moods[idx] for idx in pred_mood_idxs]
    pred_genres = [genres[idx] for idx in pred_genre_idxs]

    correct_mood = "CORRECT" if (set(pred_moods) & set(GT_moods)) else "INCORRECT"
    correct_genre = "CORRECT" if (set(pred_genres) & set(GT_genres)) else "INCORRECT"
    print("Mood: ", correct_mood, " (GT: ", GT_moods, ", Pred mood: ", pred_moods, ")")
    print("Genre: ", correct_genre, " (GT: ", GT_genres, ", Pred genre: ", pred_genres, ")\n")
    
    # Create genre-based prompt
    intersect_mood = [item for item in pred_moods if item in GT_moods]
    intersect_genre = [item for item in pred_genres if item in GT_genres]
    prompt_mood = intersect_mood[0] if bool(intersect_mood) else pred_moods[0] 
    prompt_genre =  intersect_genre[0] if bool(intersect_genre) else pred_genres[0]
    SD_prompt = prompt_mood + " " + prompt_genre + " music album cover"
    print("Input prompt: ", SD_prompt, "\n")
    text_pipe = pipeline('text-generation', model='daspartho/prompt-extend', device=0)
    extended_SD_prompt = text_pipe(SD_prompt+',', num_return_sequences=1, max_length=25)[0]["generated_text"]
    extended_SD_prompt
    print("Extended input prompt: ", extended_SD_prompt, "\n")

    # Generate output imgs with SD
    num_out_imgs = 2
    SD_model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(SD_model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(SD_model_id, scheduler=scheduler, torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing() # Reduce GPU usage
    SD_images_album = pipe([SD_prompt] * num_out_imgs).images             # Album cover
    SD_images_concept = pipe([extended_SD_prompt] * num_out_imgs).images    # Concept art
    
    # Dump outputs
    out_dir = "./CPSC532S_FinalProject/outputs/inference/"
    info_dict = {
        'URL': vid_url,
        'GT_moods': GT_moods,
        'GT_genres': GT_genres,
        'pred_moods': pred_moods, 
        'pred_genres': pred_genres,
        'album_prompt': SD_prompt,
        'concept_prompt': extended_SD_prompt,
    }
    info_json = json.dumps(info_dict)
    with open(out_dir + "info.json", "w") as outfile:
        outfile.write(info_json)
    for i, img in enumerate(SD_images_album):
        img.save(out_dir + "album_output_" + str(i) + ".png")
    for i, img in enumerate(SD_images_concept):
        img.save(out_dir + "concept_output_" + str(i) + ".png")

if __name__ == "__main__":
    main()