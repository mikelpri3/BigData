import random 
import numpy as np
import torch
from torch.utils.data import DataLoader

#Set a seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

from sklearn.model_selection import train_test_split


from load_data import brain_dataset

#Split data statifying by classes
labels = brain_dataset.labels

train_indices, test_indices = train_test_split(
    list(range(len(brain_dataset))),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

#Make the data Loaders
train_dataset = torch.utils.data.Subset(brain_dataset, train_indices)
test_dataset = torch.utils.data.Subset(brain_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

