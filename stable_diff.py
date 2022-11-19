import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline
from prompt_templates import genre_prompts, moods

# load the pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to("cuda")

# Select genre-based prompt
genre_idx = 16
mood_idx = 2
mood = moods[mood_idx]
prompt = genre_prompts[genre_idx].replace("*", mood)   # Get prompt from our list, add in the mood
print("Input prompt: ", prompt)

# Get output img
images = pipe(prompt).images
images[0].save("./CPSC532S_FinalProject/outputs/output.png")
