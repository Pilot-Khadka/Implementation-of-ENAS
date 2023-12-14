# from utils import Config
# import argparse
import torch
import torchview
import torch.nn as nn
from controller import *
from CustomArchitecture import *
from utils import *
import torch.optim as optim



# def train():
#     batch_size = 32
#     num_classes = 10
#     learning_rate = 0.001
#     epochs = 5
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = CustomModule(3,6)
#     # print(model)
#
#     # model = model.to(device)
#     criteria = nn.CrossEntropyLoss()
#
#     optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9,weight_decay=0.005 )
#
#     total_step = len(train_loader)
#
#     for epoch in range(epochs):
#         for data in train_loader:
#             images,labels = data
#             images = images.to(device)
#             labels = labels.to(device)
#
#             outputs = model(images)
#             loss = criteria(outputs,labels)
#
#             #Backprop
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#     print('Epoch [{}/{}, Loss: {:4f}]'.format(epoch+1, epochs, loss.item()))
#

if __name__ == '__main__':
    # train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    controller_model = Controller()
    model = CustomModule(3,6)
    # print(model.module_list)

    inputs = torch.zeros(1)

    print(inputs.shape)

    # inputs = inputs.to(device)

    # loss_function = nn.NLLLoss()
    # optimizer = optim.SGD(controller_model.parameters(), lr=0.1)
    # with torch.no_grad():

    pred = controller_model(inputs)

    print(pred.shape)


