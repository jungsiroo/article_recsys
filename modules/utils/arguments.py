import argparse
from types import SimpleNamespace

def get_args():
    args = argparse.ArgumentParser()

    # Hyper Params
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch', type=int, default=32)
    args.add_argument('--test_batch', type=int, default=3008)
    args.add_argument('--epoch', type=int, default=300)
    args.add_argument('--neg_sampling_ratio', type=int, default=15)
    args.add_argument('--n_latent', type=int, default=16)
    args.add_argument('--n_depth', type=int, default=4)
    args.add_argument('--n_feature', type=int, default=64)
    args.add_argument('--dropout_rate', type=float, default=0.)
    args.add_argument('--lr', type=float, default=1e-6)
    args.add_argument('--alpha', type=float, default=0.3)
    args.add_argument('--view', type=str, default='include')

    # Tools
    args.add_argument('--patience', type=int, default=20)
    args.add_argument('--top_k', type=int, default=5)
    args.add_argument('--emb_type', type=str, default='title')

    # Path / Name
    args.add_argument('--root', type=str, default='/home/elicer/recsys')
    args.add_argument('--model', type=str, default='similarity')

    config = args.parse_args()
    return config

def get_jupyter_args():
    config = {
        'seed': 42,
        'batch': 32,
        'test_batch': 3008,
        'epoch': 300,
        'neg_sampling_ratio': 15,
        'n_latent': 16,
        'n_depth': 4,
        'n_feature': 64,
        'dropout_rate': 0.0,
        'lr': 1e-6,
        'patience': 20,
        'top_k': 5,
        'emb_type': 'title',
        'root': '/home/elicer/recsys',
        'model': 'similarity'
    }
    
    config = SimpleNamespace(**config)
    return config
