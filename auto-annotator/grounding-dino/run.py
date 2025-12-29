import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class GroundingDINORun:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.model_id = "IDEA-Research/grounding-dino-base"

    def run(self):

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

        for image_path in os.listdir(self.input_folder):
            image = Image.open(os.path.join(self.input_folder, image_path)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            outputs = model(**inputs)
            
            
