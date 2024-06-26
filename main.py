from commons import logger
from data_ingestion import DownloadDataset
from model_training import ModelTraining
from model_inference import ModelInference
from torchvision import models
import torch.nn as nn

if __name__ == "__main__":
    STAGE_NAME = "Data Ingestion"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DownloadDataset()
        obj._check_json()
        obj.download_data()
        logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)

    STAGE_NAME = "Model Training"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model = models.resnet18(pretrained=True)
        model_obj = ModelTraining()
        dataset_classes, class_length , train_set_loader, val_set_loader = model_obj.data_loaders()

        num_features = model.fc.in_features
        num_classes = class_length
        model.fc = nn.Linear(num_features, num_classes)

        trained_model = model_obj.train_nn(model, train_set_loader, val_set_loader)
        model_obj.save_model(trained_model)
        logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)

    STAGE_NAME = "Model Inference"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelInference()
        obj.inference()
        logger.info(f">>>>>> stage {STAGE_NAME} Completed <<<<<<")
    except Exception as e:
        logger.exception(e)

    