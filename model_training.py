import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from commons.common_utils import set_device, data_tranforms, save_checkpoint, read_yaml
from commons.constants import CONFIG_FILE_PATH
from commons.neuralnet import NeuralNet
#from data_ingestion import DataIngestion
import os
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

@dataclass(frozen=False)        # Define Entity for Custom Data Type
class ModelParams:
    device: str
    learning_rate: float
    momentum: float
    w_decay: float
    batch_size: int
    epochs: int
    train_size: float

class ModelTraining:
    def __init__(self, config_path = CONFIG_FILE_PATH):
        try:
            self.config = ModelParams
            content = read_yaml(Path(config_path))
            content = content.model_training_config
            self.config.device = str(content.device)
            self.config.learning_rate = float(content.learning_rate)
            self.config.momentum = float(content.momentum)
            self.config.w_decay = float(content.w_decay)
            self.config.batch_size = int(content.batch_size)
            self.config.epochs = int(content.epochs)
            self.config.train_size = float(content.train_size)
        except Exception as e:
            print(e)

    def data_loaders(self):
        """
        Create DataLoader Objects in PyTorch
        Args:
            None
        Returns:
            dataset_classes: list (list of classes in the dataset)
            class_length: int (number of classes in the dataset)
            train_set_loader: torch.utils.data.DataLoader (DataLoader object)
            val_set_loader: torch.utils.data.DataLoader (DataLoader object)
        """

        data_dir = 'data/animals10/raw-img'
        image_datasets = datasets.ImageFolder(root=data_dir, transform=data_tranforms())

        dataset_size = len(image_datasets)
        self.dataset_classes = image_datasets.classes
        self.class_length = len(image_datasets.classes)

        train_size = int(dataset_size*self.config.train_size)

        train_set, val_set = torch.utils.data.random_split(image_datasets, (train_size, dataset_size-train_size))

        train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=8)
        val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config.batch_size, shuffle=True, num_workers=8)

        return self.dataset_classes, self.class_length, train_set_loader, val_set_loader

    def train_nn(self, model, train_loader, val_loader):
        """
        Model Training Function
        Args:
            model: Model to be trained
            train_loader: DataLoader for Training Set
            val_loader: DataLoader for Validation Set
            criterion: Loss Function
            optimizer: Optimizer
            n_epochs: Number of Epochs
        Returns:
            model: Trained Model
        """
        
        device = set_device(self.config.device)
        #model = NeuralNet()
        model = models.resnet50(pretrained=True)
        model = model.to(device)
        best_acc = 0
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.w_decay)

        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for data in train_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                total += labels.size(0)

                optimizer.zero_grad()

                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                loss = loss_fn(outputs, labels)

                loss.backward() # Backpropagation

                optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(labels==preds)

            epoch_loss = running_loss/len(train_loader)
            epoch_acc = 100.00 * running_corrects/total

            print(" -- Training Dataset -- Got %d out of %d images correctly. (%.3f%%). Epoch Loss: %.3f" % (running_corrects, total, epoch_acc, epoch_loss))

            test_data_acc = self.evaluate_model_test_set(model, val_loader)

            if test_data_acc > best_acc:
                best_acc = test_data_acc
                save_checkpoint(model, epoch, optimizer, best_acc, self.dataset_classes)

        print("Finished")
        return model

    def evaluate_model_test_set(self, model, val_loader):
        """
        Evaluate Model on Val Set
        Args:
            model: Model to be evaluated
            val_loader: DataLoader for Val Set
        Returns:
            epoch_acc: Accuracy of Model on Val Set
        """
        model.eval()
        predicted_correctly = 0
        total = 0
        device = set_device(self.config.device)
        
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                total += labels.size(0)
                
                outputs = model(images)
                _, preds = torch.max(outputs.data, 1)
                
                predicted_correctly += torch.sum(preds==labels).sum()
            
        epoch_acc = 100.00 * predicted_correctly/total
        print(" -- Validating Dataset -- Got %d out of %d images correctly. (%.3f%%)" % (predicted_correctly, total, epoch_acc))
        return epoch_acc
    
    def save_model(self, model):
        """
        Save the model using checkpoints at best epoch
        Args:
            model: Model to be saved
        Returns:
            None
        """
        chk = torch.load('best_checkpoint.pth.tar')
        print(" -- Saving the best model using best Checkpoint at Epoch {} --".format(chk['epoch']))
        model.load_state_dict(chk['model'].state_dict())
        torch.save(model, 'best_model.pth')
        print("Model saved successfully: {}".format('best_model.pth'))

    