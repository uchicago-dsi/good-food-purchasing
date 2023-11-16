import logging

import torch

logging.basicConfig(level=logging.INFO)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Device : {device}")



if __name__ == '__main__':
    logging.info('Starting')
