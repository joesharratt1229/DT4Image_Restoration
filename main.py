from trainer.base import MetaTrainer
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('batch_size', type = int)

args = parser.parse_args()

if __name__ == '__main__':

    trainer = MetaTrainer(batch_size = args.batch_size)
    trainer.train()




