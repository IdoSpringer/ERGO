import torch
import argparse
from joblib import Parallel, delayed
import queue
import os
import ergo_data_loader
import pickle


def main(args):
    # Parallel training of models with missing position
    iter = args.iteration
    # Define number of GPUs available
    N_GPU = torch.cuda.device_count()
    # Put indices in queue
    global q
    q = queue.Queue(maxsize=N_GPU)
    for i in range(1, N_GPU):
        q.put(i)

    max_len = 28

    # Change loop
    Parallel(n_jobs=N_GPU, backend="threading")(
        delayed(runner)(args, mis, iter) for mis in range(max_len))


def runner(args, mis, iter):
    gpu = q.get()
    # print(x, gpu)
    real_device = 'cuda:' + str(gpu)
    print(real_device)
    # Put here your job cmd
    cmd = "python -m mis_pos.train_mis %s %s %s %s %s" % (args.model_type, args.dataset, args.sampling,
                                                          str(mis), str(iter))
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))
    # return gpu id to queue
    q.put(gpu)


def sample_data(args):
    # Load data
    if args.dataset == 'mcpas':
        datafile = r'data/McPAS-TCR.csv'
    if args.dataset == 'vdjdb':
        datafile = r'data/VDJDB_complete.tsv'
    train, test = ergo_data_loader.load_data(datafile, args.dataset, args.sampling)
    dir = 'mis_pos'
    args.train_data_file = dir + '/' + '_'.join([args.dataset, 'train'])
    with open(args.train_data_file + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    args.test_data_file = dir + '/' + '_'.join([args.dataset, 'test'])
    with open(args.test_data_file + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    parser.add_argument("model_type")
    parser.add_argument("dataset")
    parser.add_argument("sampling")
    parser.add_argument("iteration")
    parser.add_argument("--ae_file")
    parser.add_argument("--train_data_file")
    parser.add_argument("--test_data_file")
    args = parser.parse_args()

    if args.function == 'train':
        main(args)
    elif args.function == 'sample':
        sample_data(args)

# nohup python mis_pos/mis_pos_scheduler.py train ae mcpas specific 3
