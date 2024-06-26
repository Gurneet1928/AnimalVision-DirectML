import torch
import torch_directml
import os
from PIL import Image
from commons.common_utils import data_tranforms, set_device

class ModelInference:
    def __init__(self):
        if not os.path.exists("best_checkpoint.pth.tar"):
            raise FileNotFoundError("Model Checkpoint Does not Exist\nPlease train the model first!!")

        if not os.path.exists("best_model.pth"):
            raise FileNotFoundError("Model not found\nPlease train the model first!!")
        
        self.model = torch.load('best_model.pth')
        self.chk = torch.load('best_checkpoint.pth.tar')

    def inference(self, img_path):
        
        classes = self.chk['classes']

        model = self.model
        model = model.eval()

        image = Image.open(img_path)
        transforms = data_tranforms()
        image = transforms(image)
        image = image.unsqueeze(0).to(set_device("dml"))

        output = model(image)
        _, pred = torch.max(output.data, 1)

        print(classes[pred])
            

if __name__ == "__main__":
    obj = ModelInference()
    obj.inference("dog.jpg")
        