import torch
import numpy as np
from diffusers import StableDiffusionPipeline

from utils.models import genre_classifier, mood_classifier
from utils.dataset import AudioSet, genre_indices, mood_indices
from utils.prompt_templates import genre_prompts, moods, genres

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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
    print("Input video ID:", vid_id)
    mood_idx = torch.argmax(mood)
    genre_idx = torch.argmax(genre)
    feature = feature.to(device)

    # Get mood/genre of feature
    pred_mood = mood_model(feature)
    pred_genre = genre_model(feature)

    pred_mood_idx = torch.argmax(pred_mood).item()
    pred_genre_idx = torch.argmax(pred_genre).item()
    pred_mood_audioset_idx = mood_indices[pred_mood_idx]
    pred_genre_audioset_idx = genre_indices[pred_genre_idx]

    print("GT mood: ", moods[mood_idx], ", Pred mood: ", moods[pred_mood_idx])
    print("GT genre: ", genres[genre_idx], ", Pred genre: ", genres[pred_genre_idx], "\n")
    
    # Create genre-based prompt
    SD_input_mood = moods[pred_mood_idx]
    SD_prompt = genre_prompts[pred_genre_idx].replace("*", SD_input_mood)   # Get prompt from our list, add in the mood
    print("Input prompt: ", SD_prompt)

    # Generate output img with SD
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to("cuda")
    SD_images = pipe(SD_prompt).images
    SD_images[0].save("./CPSC532S_FinalProject/outputs/output.png")

if __name__ == "__main__":
    main()