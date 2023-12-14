# ***************** Search Space Design ***************###
nodes = [8, 16, 32, 64, 128, 256, 512]
activations = ['Sigmoid', 'Tanh', 'ReLU']

# ***************** Sample architecture***************###
number_of_samples = 10  # -> number of architecture samples generated for an epoch
max_len = 3  # -> is max architecture length of an individual model
# -> Basically means model can have upto 10 layers
learning_rate = 0.001
validation_split = 0.33

# ***************** Dataset ***************###
target_classes = 3

# ***************** Controller Parameters***************###
controller_lstm_dim = 100
controller_input_shape = (1, max_len - 1)
samples_generated_per_controller_epoch = 3
LSTM_learning_rate = 0.001
LSTM_training_epochs = 3