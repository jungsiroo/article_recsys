import random
import torch
import pandas as pd
import numpy as np

def seed_fix(seed): #시드 고정 함수
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def display_line(desc:str):
    print("*"*30)
    print(desc)
    print()
