import os
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

class ModelHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize_models()
        return cls._instance

    def initialize_models(self):
        self.local_model_name = "C:\\Users\\91798\\Downloads\\Img_generation_captioning\\Notebooks\\local_models"
        self.local_model = VisionEncoderDecoderModel.from_pretrained(self.local_model_name)
        self.local_feature_extractor = ViTFeatureExtractor.from_pretrained(self.local_model_name)
        self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)
        self.max_length = 16
        self.num_beams = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model.to(self.device)
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

    def predict_step(self, image_paths):
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        pixel_values = self.local_feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.local_model.generate(pixel_values, **self.gen_kwargs)

        preds = self.local_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
