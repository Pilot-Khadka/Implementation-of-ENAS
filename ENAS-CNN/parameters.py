###***************** Search Space Design ***************###
kernel_size = [3,5,7]
operation = ['Conv2d','AvgPool2d','MaxPool2d']
number_of_channels = []
###***************** Sample architecture***************###
number_of_samples = 10  # -> number of architecture samples generated for an epoch
max_len = 10  # -> is max architecture length of an individual model
# -> Basically means model can have upto 10 layers
learning_rate = 0.001
validation_split = 0.33

#*******************Architecture params*************
number_of_cells = 6

#
#Controller params
#
LSTM_training_epochs = 8
LSTM_learning_rate = 0.00005