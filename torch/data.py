import musdb
import random
import numpy as np

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

def augment(x):
    gain = random.uniform(0.25,1.25)
    ch_swap = random.random() > 0.5

    if ch_swap:
        return x * gain
    else:
        return x[::-1] * gain

class MusdbDataset(Dataset):
    def __init__(self, N:int, size:int, musdb_root:str, target:str, subset:str):
        self.N = N 
        self.size = size
        self.target = target

        self.chunk_duration = self.size / 44100.0 # in seconds
        
        if subset == "train":            
            self.db = musdb.DB(root=musdb_root, subsets="train", split="train")
        elif subset == "valid":
            self.db = musdb.DB(root=musdb_root, subsets="train", split="valid")
        elif subset == "test":
            self.db = musdb.DB(root=musdb_root, subsets="test")

        if subset == "train":
            self.transform = augment
        else:
            self.transform = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        track = random.choice(self.db.tracks)
        
        track.chunk_duration = self.chunk_duration
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)

        mix_track = track.audio.astype(np.float32)
        target_track = track.targets[self.target].audio.astype(np.float32)

        if self.transform:
            mix_track = self.transform(mix_track)
            target_track = self.transform(target_track)
            
        return (
            mix_track.T,
            target_track.T
        )


class MusdbDataModule(pl.LightningDataModule):
    def __init__(self, musdb_root: str="D:/MUSDB18/Full", target: str="drums", N: int=512, input_size: int=2**17, batch_size: int=8):
        super().__init__()
        self.target = target
        self.batch_size = batch_size        
        self.input_size = input_size
        self.N = N
        self.musdb_root = musdb_root 

    def setup(self, stage):
        self.train = MusdbDataset(N=self.N, size=self.input_size, musdb_root=self.musdb_root, target=self.target, subset="train")        
        self.valid = MusdbDataset(N=self.batch_size*3, size=self.input_size, musdb_root=self.musdb_root, target=self.target, subset="valid")
        self.test = MusdbDataset(N=self.N, size=self.input_size, musdb_root=self.musdb_root, target=self.target, subset="test")

    def train_dataloader(self):        
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return self.test_dataloader()


if __name__ == "__main__":
    m = MusdbDataModule()
