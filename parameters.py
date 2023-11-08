
###***************** Search Space Design ***************###
nodes = [8, 16, 32, 64, 128, 256, 512]
activations = ['sigmoid', 'tanh', 'relu']

###***************** Sample architecture***************###
number_of_samples = 10 #-> number of architecture samples generated for an epoch
max_len = 10 #-> is max architecture length of an individual model
            #-> Basically means model can have upto 10 layers


###***************** Dataset ***************###
target_classes = 3

###***************** Controllet Parameters***************###
controller_lstm_dim = 100
controller_batch_size = 32
controller_input_shape = (1, max_len-1)
