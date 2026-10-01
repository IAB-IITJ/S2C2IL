import os
import torch

gpu = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # use gpu0,1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu

ssl_task = True 

dataset = 'svhn'
random_state = 111
num_channels = 3
image_shape = 32

#######################################################################
# # SSL parameters 

pretext_task = 'label_augmentation'

lr = 0.0001
batch_size = 64
w_d = 1e-5
momentum = 0.9


alpha = 10 #0.01    #0, 2
beta = 18 # 0.1  #0

num_epochs  = 20
num_tasks   = 3   
num_classes = 2   

random_seed_list = [8, 69, 78, 121, 512]
random_seed_counter = 0
#########################################################################
# # OWM parameters

nepochs = [25, 25, 25, 25, 25]
owm_lrs = [0.1, 0.1, 0.1, 0.1, 0.1]

num_layers = 3
model = 'owm_'+str(num_layers)

pretrained = True
#pretrained = False
