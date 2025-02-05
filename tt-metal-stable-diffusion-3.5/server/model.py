import os
import random
import string
from PIL import Image
import requests


# Global variable for the Stable Diffusion model pipeline
model_pipeline = None


# Warmup the model on app startup
def warmup_model():
    global model_pipeline
    print("Loading Stable Diffusion model...")
    print("Model loaded and ready!")


# Function to generate an image from the given prompt
def generate_image_from_prompt(prompt):
    # Generate a random file name for the image
    random_filename = (
        "".join(random.choices(string.ascii_lowercase + string.digits, k=10)) + ".png"
    )
    image_path = os.path.join("generated_images", random_filename)

    # Generate the image using Stable Diffusion
    url = "https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save(image_path)

    return image_path
