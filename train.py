import sys
import warnings
import argparse
import pandas as pd
import numpy as np
from modules.utils import *
from modules.models import *
from modules.data import *
from sklearn.metrics.pairwise import cosine_similarity

def run(config):
    display_line('Load Data...')
    data_storage = DataStorage(config)
    train_data = data_storage.load_train_dataset()

    model = build_model(config, train_data)
    display_line(f'Model Build Sucess : {config.model.title()}')

    result = model.fit()
    result = data_storage.postprocess(result)

    display_line('Train Done & Saving...')
    save_submission_file(config, result)


if __name__ == "__main__":
    config = get_args()
    warnings.filterwarnings('ignore')

    seed_fix(config.seed)
    run(config)