import time
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support

class CNN(nn.Module):
    """Convolutional Neural Network."""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1: 3 x 50 x 50 -> 8 x 48 x 48
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # Conv Layer block 2: 8 x 48 x 48 -> 8 x 46 x 46
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1), 
            nn.BatchNorm2d(8), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # Conv Layer block 2: 8 x 46 x 46 -> 8 x 44 x 44
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # Conv Layer block 3: 8 x 44 x 44 -> 8 x 42 x 42
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * 42 * 42, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
    
def train_and_validate(model, train_loader, validation_loader, loss_function, optimizer, epochs, device):
    """train and validate"""
    
    train_losses = []
    train_scores = []
    train_recalls = []
    train_precisions = []
    train_accuracies = []
    
    validation_losses = []
    validation_scores = []
    validation_recalls = []
    validation_precisions = []
    validation_accuracies = []

    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        all_train_labels = []
        all_train_predictions = []
        train_loss, correct_train, total_train = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            predictions = torch.max(outputs, 1)[1]
            total_train += labels.size(0)
            correct_train += (predictions == labels).sum().item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_predictions.extend(predictions.cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        train_precision, train_recall, train_score, train_support = precision_recall_fscore_support(all_train_labels, all_train_predictions, average='macro')
        
        train_losses.append(train_loss)
        train_scores.append(train_score)
        train_recalls.append(train_recall)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)

        # Validation Phase
        model.eval()
        all_validation_labels = []
        all_validation_predictions = []
        validation_loss, correct_validation, total_validation = 0, 0, 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = loss_function(outputs, labels)
                validation_loss += loss.item() * images.size(0)
                
                predictions = torch.max(outputs, 1)[1]
                total_validation += labels.size(0)
                correct_validation += (predictions == labels).sum().item()

                all_validation_labels.extend(labels.cpu().numpy())
                all_validation_predictions.extend(predictions.cpu().numpy())
        
        validation_loss /= len(validation_loader.dataset)
        validation_accuracy = correct_validation / total_validation
        validation_precision, validation_recall, validation_score, validation_support = precision_recall_fscore_support(all_validation_labels, all_validation_predictions, average='macro')

        validation_losses.append(validation_loss)
        validation_scores.append(validation_score)
        validation_recalls.append(validation_recall)
        validation_accuracies.append(validation_accuracy)
        validation_precisions.append(validation_precision)
        
        end_time = time.time()
        print(f'Epoch [{epoch + 1} / {epochs}], time: {(end_time - start_time) / 60:.2f} minutes')
        print(f'Train metrics - losses: {train_loss:.4f}, scores: {train_score:.4f}, recalls: {train_recall:.4f}, precisions: {train_precision:.4f}, accuracies: {train_accuracy:.4f}')
        print(f'Validation metrics - losses: {validation_loss:.4f}, scores: {validation_score:.4f}, recalls: {validation_recall:.4f}, precisions: {validation_precision:.4f}, accuracies: {validation_accuracy:.4f}')
    
    # Store metrics
    train_metrics = {
        'losses': train_losses,
        'scores': train_scores, 
        'recalls': train_recalls,
        'precisions': train_precisions,
        'accuracies': train_accuracies,
    }

    validation_metrics = {
        'losses': validation_losses,
        'scores': validation_scores, 
        'recalls': validation_recalls,
        'precisions': validation_precisions,
        'accuracies': validation_accuracies,
    }
    
    return train_metrics, validation_metrics

def predict(model, loader, device, loss_function):
    model.eval()
    all_labels = []
    all_predictions = []
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss += loss_function(outputs, labels).item() * images.size(0)
            
            total += labels.size(0)
            predictions = torch.max(outputs, 1)[1]
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    loss /= len(loader.dataset)
    accuracy = correct / total
    precision, recall, score, support = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    
    metrics = {
        'loss': loss,
        'score': score, 
        'recall': recall,
        'accuracy': accuracy,
        'precision': precision,
    }
    return metrics
