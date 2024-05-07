import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

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

def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)

def value(loader, device):
    mean = 0.0
    variance = 0.0
    number_ones = 0
    number_zeros = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        images = images.view(images.size(0), images.size(1), -1)
        variance += images.var(2).sum(0)
        mean += images.mean(2).sum(0) 

        number_ones += torch.sum(labels == 1).item()
        number_zeros += torch.sum(labels == 0).item()

    number = number_ones + number_zeros
    weight_ones = 1 / number_ones * number / 2.0
    weight_zeros = 1 / number_zeros * number / 2.0
    weight = torch.tensor([weight_zeros, weight_ones], dtype=torch.float32).to(device)
    
    mean /= number
    variance /= number
    return weight, mean, variance

def dataset(data, device, batch_size, worker_size):
    """Returns the training and testing datasets."""
    train_data, evaluation_data = train_test_split(data, test_size=0.2, random_state=0)
    validation_data, test_data = train_test_split(evaluation_data, test_size=0.5, random_state=0)
    
    initial_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((50, 50), antialias=True)])
    
    initial_dataset = ImageDataset(dataframe=train_data, transform=initial_transform)
    initial_loader = DataLoader(dataset=initial_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=worker_size, worker_init_fn=worker_init_fn)

    weight, mean, variance = value(initial_loader, device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((50, 50), antialias=True), transforms.Normalize(mean, variance.sqrt())])
    
    test_dataset = ImageDataset(dataframe=test_data, transform=transform)
    train_dataset = ImageDataset(dataframe=train_data, transform=transform)
    validation_dataset = ImageDataset(dataframe=validation_data, transform=transform)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=worker_size, worker_init_fn=worker_init_fn)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=worker_size, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=worker_size, worker_init_fn=worker_init_fn)
    
    return weight, train_loader, validation_loader, test_loader