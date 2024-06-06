import argparse

def get_args():
    args = argparse.ArgumentParser()

    args.add_argument('--save_path', type=str, default="./submissions")
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--model', type=str, default='baseline')

    config = args.parse_args()
    return config