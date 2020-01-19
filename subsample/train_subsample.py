import torch
import argparse
import lstm_utils as lstm
import ae_utils as ae
import pickle
import os


def ae_get_lists_from_pairs(pairs, max_len):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        if len(tcr) >= max_len:
            continue
        tcrs.append(tcr)
        peps.append(pep[0])
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


def lstm_get_lists_from_pairs(pairs):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label = pair
        tcrs.append(tcr)
        peps.append(pep[0])
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


def train_model(args):
    t_samples = int(args.t_samples)
    iter = args.iteration

    # hyper-params
    params = {}
    params['lr'] = 1e-4
    params['wd'] = 0
    params['epochs'] = 100
    params['batch_size'] = 50
    params['lstm_dim'] = 500
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0
    params['enc_dim'] = 100
    params['train_ae'] = True

    arg = {}
    arg['siamese'] = False
    arg['ae_file'] = args.ae_file
    arg['train_auc_file'] = 'ignore'
    arg['test_auc_file'] = 'subsample/subsample_auc/' + '_'.join([args.model_type, args.dataset,
                                                              'test', str(iter), str(t_samples)])
    if os.path.isfile(arg['test_auc_file']):
        # already done
        exit()

    # Load autoencoder params
    if args.model_type == 'ae':
        args.ae_file = 'TCR_Autoencoder/tcr_ae_dim_' + str(params['enc_dim']) + '.pt'
        checkpoint = torch.load(args.ae_file, map_location='cuda:0')
        params['max_len'] = checkpoint['max_len']
        params['batch_size'] = checkpoint['batch_size']
    arg['ae_file'] = args.ae_file

    # Load data
    dir = 'subsample'
    args.train_data_file = dir + '/' + '_'.join([args.dataset, 'train.pickle'])
    args.test_data_file = dir + '/' + '_'.join([args.dataset, 'test.pickle'])
    # Read train data
    with open(args.train_data_file, "rb") as file:
        train = pickle.load(file)
    # Read test data
    with open(args.test_data_file, "rb") as file:
        test = pickle.load(file)

    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    if args.model_type == 'lstm':
        amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    if args.model_type == 'ae':
        pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
        tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    device = 'cuda:0'
    if args.model_type == 'ae':
        # train
        train_tcrs, train_peps, train_signs = ae_get_lists_from_pairs(train, params['max_len'])
        stop = t_samples * 10000
        train_tcrs, train_peps, train_signs = train_tcrs[:stop], train_peps[:stop], train_signs[:stop]
        train_batches = ae.get_batches(train_tcrs, train_peps, train_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])
        # test
        test_tcrs, test_peps, test_signs = ae_get_lists_from_pairs(test, params['max_len'])
        test_batches = ae.get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])
        # Train the model
        model, best_auc, best_roc = ae.train_model(train_batches, test_batches, device, arg, params)
        pass
    if args.model_type == 'lstm':
        # train
        train_tcrs, train_peps, train_signs = lstm_get_lists_from_pairs(train)
        stop = t_samples * 10000
        train_tcrs, train_peps, train_signs = train_tcrs[:stop], train_peps[:stop], train_signs[:stop]
        lstm.convert_data(train_tcrs, train_peps, amino_to_ix)
        train_batches = lstm.get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])
        # test
        test_tcrs, test_peps, test_signs = lstm_get_lists_from_pairs(test)
        lstm.convert_data(test_tcrs, test_peps, amino_to_ix)
        test_batches = lstm.get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])
        # Train the model
        model, best_auc, best_roc = lstm.train_model(train_batches, test_batches, device, arg, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("dataset")
    parser.add_argument("sampling")
    parser.add_argument("t_samples")
    parser.add_argument("iteration")
    parser.add_argument("--ae_file")
    parser.add_argument("--train_data_file")
    parser.add_argument("--test_data_file")
    args = parser.parse_args()
    train_model(args)



