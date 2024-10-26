import random
import torch
import pandas as pd
import numpy as np
import re
import string
import glob
import pickle
import os 

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

def save_submission_file(config, data):
    data.to_csv(f"{config.root}/submissions/{config.model}_{config.alpha}_{config.view[0]}.csv", index=False)

    display_line(f'{config.model.upper()} model submission file saved!')