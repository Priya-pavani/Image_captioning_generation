import requests
from PIL import Image
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": "Bearer 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"}



def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

print(query('Flicker8k_Dataset\\17273391_55cfc7d3d4.jpg'))
