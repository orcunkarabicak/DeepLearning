from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (Pandas DataFrame): DataFrame containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 1]  # Access image path
        image = Image.open(img_path).convert('RGB')  # Load image and ensure RGB
        label = self.dataframe.iloc[idx, 2]  # Access the label

        if self.transform:
            image = self.transform(image)

        return image, label
