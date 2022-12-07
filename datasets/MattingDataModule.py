import pandas
import os
import pytorch_lightning
from torch.utils.data import DataLoader, Subset
from datasets.MattingDataset import MattingDataset
import numpy
import torch
from tqdm import tqdm
from utils.dataset import generate_dataframe
from utils.data import Data
from sklearn.model_selection import train_test_split

def split_dataset(df):
    return numpy.split(df.sample(frac=1, random_state=42), [int(0*len(df)), int(1*len(df))])

class MattingDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        dataset_name = None,
        batch_size: int = None,
        num_workers: int = None,
        transform = None,
    ):  
        if dataset_name == None:
            raise Exception("Dataset not given")
        self.data = Data(dataset_name)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        

    def prepare_data(self) -> None:
        self.data.load()

    def setup(self, stage: str = None) -> None:
        df = self.data.dataframe()
        train_df, test_df, val_df = split_dataset(df)
        self.train_data = MattingDataset(train_df)
        self.val_data = MattingDataset(val_df, train=False, transform=self.transform)
        self.test_data = MattingDataset(test_df, train=False, transform=self.transform)

        

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
