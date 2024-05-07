from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
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
        img_path = self.dataframe.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx, 2]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label