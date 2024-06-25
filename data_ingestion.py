import os
from pathlib import Path
from dataclasses import dataclass
from box import ConfigBox
import yaml
import opendatasets as od
from commons.common_utils import read_yaml
from commons.constants import CONFIG_FILE_PATH

@dataclass(frozen=False)         # Define Entity for Custom Data Type
class DataIngestion:
    download_url: str
    local_data_file: Path
    images_loc: Path

class DownloadDataset:
    def __init__(self, path_to_config: Path = CONFIG_FILE_PATH):
        try:
            self.config = DataIngestion
            content = read_yaml(Path(path_to_config))
            self.config.download_url = str(content.data_ingestion.download_url)
            self.config.local_data_file = Path(content.data_ingestion.local_data_file)
            self.config.images_loc = Path(content.data_ingestion.images_loc)
        except Exception as e: 
            print(e)

    def _check_json(self):
        """
        Check the kaggle.json file
        Args:
            None
        Returns:
            Nothing"""
        if not os.path.exists("kaggle.json"):
            raise FileNotFoundError("kaggle.json file not found in the current directory")
            

    def download_data(self):
        """
        Download the dataset from the source URL and save it to the local directory.
        Args:
            None
        Returns:
            Nothing
        """
        if not os.path.exists(self.config.local_data_file):
            os.makedirs(self.config.local_data_file)
            print("Data directory created")
        else:
            print("Data directory already exists")

        if not os.path.exists(self.config.images_loc):
            od.download_kaggle_dataset(self.config.download_url, self.config.local_data_file)
            print("Data Downloaded Successfully")
        else:
            print("Data already downloaded")

random_var = DataIngestion("","","")
obj = DownloadDataset()
obj._check_json()
obj.download_data()