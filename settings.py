import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EP = 20
MEMORY_CAPACITY = 90000
BATCH_SIZE = 128
LEARNING_RATE = 0.00025
GAMMA = 0.999
EPSILON_START = 1.0
EPSILON_FINAL = 0.1
EPSILON_DECAY = 1000000
TARGET_STEP = 10000
SAVE_DIR = "./model"
LOG_FILE = "./log.txt"
LOAD_MODEL = False