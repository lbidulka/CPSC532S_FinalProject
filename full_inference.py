import torch
import numpy as np
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import pipeline
import lovely_tensors as lt
lt.monkey_patch()
import json

from utils.models import genre_classifier, mood_classifier
from utils.dataset import AudioSet, genre_indices, mood_indices, moods, genres
# from utils.label_info import moods, genres

# SEED = 1234
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load classifiers
    checkpoint_path = "./CPSC532S_FinalProject/checkpoints/"
    mood_model = torch.load(checkpoint_path + "mood_classifier.pth")
    genre_model = torch.load(checkpoint_path + "genre_classifier.pth")

    # Get extract a sample
    print("Loading dataset...", end=" ")
    audioset = AudioSet(batch_size=1, inference=True)
    print("done!\n")
    batch = next(iter(audioset.inf_testloader))
    feature, genre, mood, vid_id, times = batch

    vid_url = ""
    for id in vid_id[0].tolist():
        vid_url += chr(int(id))
    vid_url = "https://www.youtube.com/watch?v=" + vid_url + "&t=" + str(int(times[0,0].item())) + "s"
    print("Input video URL: ", vid_url)
    print("Time from {}s to {}s\n".format(times[0,0].item(), times[0,1].item()))

    mood_idx = torch.argmax(mood)
    genre_idx = torch.argmax(genre)
    feature = feature.to(device)

    # Get mood/genre of feature
    pred_mood = mood_model(feature)
    pred_genre = genre_model(feature)
    pred_mood_idx = torch.argmax(pred_mood).item()
    pred_genre_idx = torch.argmax(pred_genre).item()
    print("GT mood: ", moods[mood_idx], ", Pred mood: ", moods[pred_mood_idx])
    print("GT genre: ", genres[genre_idx], ", Pred genre: ", genres[pred_genre_idx], "\n")
    
    # Create genre-based prompt
    SD_prompt = moods[pred_mood_idx] + " " + genres[pred_genre_idx] + " music album cover"
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
    out_dir = "./CPSC532S_FinalProject/outputs/"
    info_dict = {
        'URL': vid_url,
        'GT_mood': moods[mood_idx],
        'GT_genre': genres[genre_idx],
        'pred_mood': moods[pred_mood_idx], 
        'pred_genre': genres[pred_genre_idx],
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