import os
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch
from PIL import Image

global local_model_name, local_model, local_feature_extractor, local_tokenizer, max_length, num_beams, gen_kwargs
global device
def initialising():
    global local_model_name, local_model, local_feature_extractor, local_tokenizer, max_length, num_beams, gen_kwargs
    global device
    local_model_name = "C:\\Users\\91798\\Downloads\\Img_generation_captioning\\Notebooks\\local_models"
    local_model = VisionEncoderDecoderModel.from_pretrained(local_model_name)
    local_feature_extractor = ViTFeatureExtractor.from_pretrained(local_model_name)
    local_tokenizer = AutoTokenizer.from_pretrained(local_model_name)
    max_length = 16
    num_beams = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    #return local_model, local_feature_extractor, local_tokenizer, gen_kwargs

def predict_step(image_paths):
    initialising()
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = local_feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = local_model.generate(pixel_values, **gen_kwargs)

    preds = local_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

print(predict_step(["C:\\Users\\91798\\Downloads\\Img_generation_captioning\\Flicker8k_Dataset\\23445819_3a458716c1.jpg",]))