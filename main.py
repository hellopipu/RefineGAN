import argparse
from Solver import Solver


def main(args):
    print(args)
    solver = Solver(args)
    if args.mode == 'train':
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'],
                        help='mode for the program')
    parser.add_argument('--mask_type', default='radial', choices=['radial', 'cartes', 'gauss', 'spiral'],
                        help='mask type')
    parser.add_argument('--sampling_rate', type=int, default=10,
                        help='sampling rate for mask, only 10, 20,30,40, ...')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size, 4,8,16, ...')
    parser.add_argument('--train_path', default='data/brain/db_train/',
                        help='train_path')
    parser.add_argument('--val_path', default='data/brain/db_valid/',
                        help='val_path')
    parser.add_argument('--test_path', default='data/brain/db_valid/',
                        help='test_path')

    args = parser.parse_args()

    main(args)
