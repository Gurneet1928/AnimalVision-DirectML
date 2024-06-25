import torch
import torch_directml
from torchvision import transforms
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
import yaml

def read_yaml(path_to_yaml: Path) -> ConfigBox:                 # Input Arguments -> Output Argument Type
    """
    Reads yaml file and returns
    Args: 
        path_to_yaml: Path input
    Raises: 
        ValueError: If file is empty
        e: empty file
    Returns: 
        ConfigBox: ConfigBox Type
    """
    
    try:
        with open(path_to_yaml, 'r') as file:
            content= ConfigBox(yaml.safe_load(file))
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Empty file: {path_to_yaml}") 
    except Exception as e: 
        return e

def data_tranforms():
    """
    Apply transformations on the dataset
    Args:
        None
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    transformation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transformation


def set_device(device: str):
    """
    Select the Device based on input
    Args:
        device: str (device name)
    Returns:
        torch.device: Device object
        
    """
    if torch_directml.device_count() > 0 and device=="dml":
        return torch_directml.device(torch_directml.default_device())
    
    '''elif torch.cuda.is_available() and device=="cuda":
        return torch.device("cuda")'''
    return torch.device("cpu")
    

def save_checkpoint(model, epoch, optimizer, best_acc):
    """
    Function to Save Model Checkpoint 
    Args:
        model: Model to be saved
        epoch: Epoch Number
        optimizer: Optimizer State
        best_acc: Best Accuracy
    """
    state = {
        'epoch': epoch + 1,
        'model': model,
        'best_accuracy': best_acc,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, 'best_checkpoint.pth.tar')