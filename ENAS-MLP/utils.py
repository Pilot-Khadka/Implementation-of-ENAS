from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from parameters import *


def data_loader(x,y,batch_size = 32):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader
