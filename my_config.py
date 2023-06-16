import torch
#General config
if torch.has_mps:
    DEVICE = "mps"
else:
    DEVICE = "cpu"
    
#Hyperparameters config
EPOCHS = 100
LEARNING_RATE = 2.25e-05
BETA_1 = 0.5
BETA_2 = 0.9
BATCH_SIZE = 6
REC_LOSS_FACTOR = 1
PER_LOSS_FACTOR = 1

#VQ-VAE config
LATENT_DIM = 256
VECTORS = 1024
BETA = 0.25

#Discriminator config
DISC_START = 1000
DISC_FACTOR = 1

#Input config
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
DATASET_PATH = "data/training"