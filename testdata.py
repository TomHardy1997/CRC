import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms  


class TestCustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        df = pd.read_csv(df)
        self.df = df
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, path

if __name__ == "__main__":
    df = 'test.csv'
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = TestCustomDataset(df,transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=135, shuffle=False, num_workers=0)
    import ipdb;ipdb.set_trace()

    