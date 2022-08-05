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

class MattingDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = Data()

    def prepare_data(self) -> None:
        self.data.load()

    def setup(self, stage: str = None) -> None:
        df = self.data.generate_dataframe()
        train_df, test_df, val_df = numpy.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
        self.train_data = MattingDataset(train_df)
        self.val_data = MattingDataset(val_df, train=False)
        self.test_data = MattingDataset(test_df, train=False)

        

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
