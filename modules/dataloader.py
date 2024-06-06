import pandas as pd
import numpy as np

def load_data():
    view_log = pd.read_csv('/home/elicer/recsys/data/view_log.csv')
    sample_sub = pd.read_csv('/home/elicer/recsys/data/sample_submission.csv')
    article = pd.read_csv('/home/elicer/recsys/data/article_info.csv')

    return view_log, article, sample_sub
