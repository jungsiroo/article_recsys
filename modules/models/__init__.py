from .similarity import *
from .npmi_cf import *

def build_model(config, dataset):
    if config.model == 'similarity':
        model = SimModel(config, *dataset)
    elif config.model == "npmi":
        model = NPMI_CF(config, *dataset)
    
    return model

