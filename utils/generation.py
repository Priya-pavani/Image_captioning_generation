import requests

API_TOKEN = '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'
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def give_prompt(inp):
	return get_output(inp)
	
	
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content
# get_prompt("Car racing on a mountain")


# You can access the image with PIL.Image for example
import io
from PIL import Image
def get_output(prompt):
	image_bytes = query({
	"inputs": prompt,
})
	image = Image.open(io.BytesIO(image_bytes))
	return image