import torch
import torch_directml
import os
from PIL import Image
from commons.common_utils import data_tranforms, set_device
from commons.common_utils import read_yaml
from commons.constants import CONFIG_FILE_PATH
from pathlib import Path
from commons import logger


class ModelInference:
    def __init__(self, path_to_config = CONFIG_FILE_PATH):
        if not os.path.exists("best_checkpoint.pth.tar"):
            logger.info("Model Checkpoint Does not Exist\nPlease train the model first!!")
            raise FileNotFoundError()

        if not os.path.exists("best_model.pth"):
            logger.info("Model Does not Exist\nPlease train the model first!!")
            raise FileNotFoundError()
        
        self.model = torch.load('best_model.pth')
        self.chk = torch.load('best_checkpoint.pth.tar')
        content = read_yaml(Path(path_to_config))
        self.file = Path(content.model_inference_file.file_path)


    def inference(self):
        
        classes = self.chk['classes']

        model = self.model
        model = model.eval()

        image = Image.open(self.file)
        transforms = data_tranforms()
        image = transforms(image)
        image = image.unsqueeze(0).to(set_device("dml"))

        output = model(image)
        _, pred = torch.max(output.data, 1)

        logger.info("Predicted Class for Image {} is {}".format(self.file, classes[pred.item()]))
            

        