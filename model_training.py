import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from commons.common_utils import set_device, data_tranforms, save_checkpoint, read_yaml
from commons.constants import CONFIG_FILE_PATH
#from data_ingestion import DataIngestion
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=False)
class ModelParams:
    device: str
    learning_rate: float
    momentum: float
    w_decay: float
    batch_size: int


class ModelTrainig:
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
        except Exception as e:
            print(e)

    def data_loaders(self, val_size: float):
        """
        Create DataLoader Objects in PyTorch
        Args:
            val_size: float (Size of the Val Set)
        Returns:
            dataset_classes: list (list of classes in the dataset)
            train_set_loader: torch.utils.data.DataLoader (DataLoader object)
            val_set_loader: torch.utils.data.DataLoader (DataLoader object)
        """

        data_dir = 'data/animals10/raw-img'
        image_datasets = datasets.ImageFolder(root=data_dir, transform=data_tranforms())

        dataset_size = len(image_datasets)
        dataset_classes = image_datasets.classes
        class_length = len(image_datasets.classes)

        train_size = int(dataset_size*val_size)

        train_set, val_set = torch.utils.data.random_split(image_datasets, (train_size, dataset_size-train_size))

        train_set_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=8)
        val_set_loader = torch.utils.data.DataLoader(val_set, batch_size=self.config.batch_size, shuffle=True, num_workers=8)

        return dataset_classes, class_length, train_set_loader, val_set_loader
    

    def train_nn(self, model, train_loader, val_loader, criterion, optimizer, n_epochs):
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
        best_acc = 0

        for epoch in range(n_epochs):
            print(f"Epoch {epoch}/{n_epochs}")
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
                loss = criterion(outputs, labels)

                loss.backward() # Backpropagation

                optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(labels==preds)

            epoch_loss = running_loss/len(train_loader)
            epoch_acc = 100.00 * running_corrects/total

            print(" -- Training Dataset -- Got %d out of %d images correctly. (%.3f%%). Epoch Loss: %.3f" % (running_corrects, total, epoch_acc, epoch_loss))

            test_data_acc = evaluate_model_test_set(model, val_loader)

            if test_data_acc > best_acc:
                best_acc = test_data_acc
                save_checkpoint(model, epoch, optimizer, best_acc)

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

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    model_obj = ModelTrainig()

    dataset_classes, class_length , train_set_loader, val_set_loader = model_obj.data_loaders(val_size = 0.8)

    num_features = model.fc.in_features
    num_classes = class_length
    model.fc = nn.Linear(num_features, num_classes)

    device = set_device("dml")
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
    n_epochs = 10

    trained_model = model_obj.train_nn(model, train_set_loader, val_set_loader, loss_fn, optimizer, n_epochs)