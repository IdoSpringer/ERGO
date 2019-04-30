import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from random import shuffle
import time
import numpy as np
import torch.autograd as autograd
from autoencoder_model import PaddingAutoencoder
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import os
import csv
import pair_sampling.load_data as d
import load_with_tcrgp as d2
import load_netTCR_data as d3
from tcr_ae_pep_lstm_model import AutoencoderLSTMClassifier
import matplotlib.pyplot as plt

def get_lists_from_pairs(pairs, max_len):
    tcrs = []
    peps = []
    signs = []
    for pair in pairs:
        tcr, pep, label, weight = pair
        if len(tcr) > max_len:
            continue
        tcrs.append(tcr)
        peps.append(pep)
        if label == 'p':
            signs.append(1.0)
        elif label == 'n':
            signs.append(0.0)
    return tcrs, peps, signs


# tcrs must have 21 one-hot, not 22. padding index in pep must be 0.
def convert_data(tcrs, peps, tcr_atox, pep_atox, max_len):
    for i in range(len(tcrs)):
        tcrs[i] = pad_tcr(tcrs[i], tcr_atox, max_len)
    convert_peps(peps, pep_atox)


def pad_tcr(tcr, amino_to_ix, max_length):
    padding = torch.zeros(max_length, 20 + 1)
    tcr = tcr + 'X'
    for i in range(len(tcr)):
        amino = tcr[i]
        padding[i][amino_to_ix[amino]] = 1
    return padding


def convert_peps(peps, amino_to_ix):
    for i in range(len(peps)):
        peps[i] = [amino_to_ix[amino] for amino in peps[i]]


def pad_batch(seqs):
    """
    Pad a batch of sequences (part of the way to use RNN batching in PyTorch)
    """
    # Tensor of sequences lengths
    lengths = torch.LongTensor([len(seq) for seq in seqs])
    # The padding index is 0
    # Batch dimensions is number of sequences * maximum sequence length
    longest_seq = max(lengths)
    batch_size = len(seqs)
    # Pad the sequences. Start with zeros and then fill the true sequence
    padded_seqs = autograd.Variable(torch.zeros((batch_size, longest_seq))).long()
    for i, seq_len in enumerate(lengths):
        seq = seqs[i]
        padded_seqs[i, 0:seq_len] = torch.LongTensor(seq[:seq_len])
    # Return padded batch and the true lengths
    return padded_seqs, lengths


def get_batches(tcrs, peps, signs, tcr_atox, pep_atox, batch_size, max_length):
    """
    Get batches from the data
    """
    # Initialization
    batches = []
    index = 0
    convert_data(tcrs, peps, tcr_atox, pep_atox, max_length)
    # Go over all data
    while index < len(tcrs) // batch_size * batch_size:
        # Get batch sequences and math tags
        # Add batch to list
        batch_tcrs = tcrs[index:index + batch_size]
        tcr_tensor = torch.zeros((batch_size, max_length, 21))
        for i in range(batch_size):
            tcr_tensor[i] = batch_tcrs[i]
        batch_peps = peps[index:index + batch_size]
        batch_signs = signs[index:index + batch_size]
        padded_peps, pep_lens = pad_batch(batch_peps)
        batches.append((tcr_tensor, padded_peps, pep_lens, batch_signs))
        # Update index
        index += batch_size
    # Return list of all batches
    return batches


def train_epoch(batches, model, loss_function, optimizer, device):
    model.train()
    shuffle(batches)
    total_loss = 0
    for batch in batches:
        tcrs, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        # print(tcrs)
        tcrs = tcrs.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        batch_signs = torch.tensor(batch_signs).to(device)
        model.zero_grad()
        probs = model(tcrs, padded_peps, pep_lens)
        # print(probs, batch_signs)
        # Compute loss
        loss = loss_function(probs, batch_signs)
        # with open(sys.argv[1], 'a+') as loss_file:
        #    loss_file.write(str(loss.item()) + '\n')
        # Update model weights
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print('current loss:', loss.item())
        # print(probs, batch_signs)
    # Return average loss
    return total_loss / len(batches)


def train_model(batches, test_batches, device, args, params):
    """
    Train and evaluate the model
    """
    losses = []
    # We use Binary-Cross-Entropy loss
    loss_function = nn.BCELoss()
    # Set model with relevant parameters
    model = AutoencoderLSTMClassifier(params['emb_dim'], device, params['max_len'], 21, params['enc_dim'], params['batch_size'], args['ae_file'], params['train_ae'])
    # Move to GPU
    model.to(device)
    # We use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    # Train several epochs
    best_auc = 0
    best_roc = None
    for epoch in range(params['epochs']):
        print('epoch:', epoch + 1)
        epoch_time = time.time()
        # Train model and get loss
        loss = train_epoch(batches, model, loss_function, optimizer, device)
        losses.append(loss)
        # Compute auc
        train_auc = evaluate(model, batches, device)[0]
        print('train auc:', train_auc)
        with open(args['train_auc_file'], 'a+') as file:
            file.write(str(train_auc) + '\n')
        test_auc, roc = evaluate(model, test_batches, device)
        if test_auc > best_auc:
            best_auc = test_auc
            best_roc = roc
        # print(roc)
        # plt.plot(roc[0], roc[1])
        # plt.show()
        print('test auc:', test_auc)
        with open(args['test_auc_file'], 'a+') as file:
            file.write(str(test_auc) + '\n')
        print('one epoch time:', time.time() - epoch_time)
    return model, best_auc, best_roc


def evaluate(model, batches, device):
    model.eval()
    true = []
    scores = []
    shuffle(batches)
    for batch in batches:
        tcrs, padded_peps, pep_lens, batch_signs = batch
        # Move to GPU
        tcrs = torch.tensor(tcrs).to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(tcrs, padded_peps, pep_lens)
        true.extend(np.array(batch_signs).astype(int))
        scores.extend(probs.cpu().data.numpy())
    # Return auc score
    auc = roc_auc_score(true, scores)
    fpr, tpr, thresholds = roc_curve(true, scores)
    return auc, (fpr, tpr, thresholds)


def main(argv):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    # Set all parameters and program arguments
    device = argv[3]
    args = {}
    args['train_auc_file'] = argv[4]
    args['test_auc_file'] = argv[5]
    args['roc_file'] = argv[6]
    args['ae_file'] = argv[1]
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 1e-5
    params['epochs'] = 200
    params['emb_dim'] = 10
    params['enc_dim'] = 30
    params['dropout'] = 0.1
    params['train_ae'] = True
    # Load autoencoder params
    checkpoint = torch.load(args['ae_file'])
    params['max_len'] = checkpoint['max_len']
    params['batch_size'] = checkpoint['batch_size']

    # Load data
    pairs_file = 'pair_sampling/pairs_data/weizmann_pairs.txt'
    if argv[-1] == 'cancer':
        pairs_file = 'pair_sampling/pairs_data/cancer_pairs.txt'
    if argv[-1] == 'shugay':
        pairs_file = 'pair_sampling/pairs_data/shugay_pairs.txt'
    if argv[-1] == 'ex_cancer':
        pairs_file = 'extended_cancer_pairs.txt'
    if argv[-1] == 'exs_cancer':
        pairs_file = 'safe_extended_cancer_pairs.txt'
    if argv[-1] == 'exnos_cancer':
        pairs_file = 'no_shugay_extended_cancer_pairs.txt'

    train, test = d.load_data(pairs_file)
    if argv[7] == 'tcrgp':
        train, test = d2.load_data(pairs_file)
    if argv[7] == 'nettcr':
        pairs_file = 'netTCR/parameters/iedb_mira_pos_uniq.txt'
        train, test = d3.load_data(pairs_file)

    # train
    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train, params['max_len'])
    train_batches = get_batches(train_tcrs, train_peps, train_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])

    # test
    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test, params['max_len'])
    test_batches = get_batches(test_tcrs, test_peps, test_signs, tcr_atox, pep_atox, params['batch_size'], params['max_len'])

    # Train the model
    model, best_auc, best_roc = train_model(train_batches, test_batches, device, args, params)

    # Save trained model
    torch.save({
                'model_state_dict': model.state_dict(),
                }, argv[2])
    # Save best ROC curve and AUC
    np.savez(args['roc_file'], fpr=best_roc[0], tpr=best_roc[1], auc=np.array(best_auc))

    pass


# python tcr_ae_pep_lstm_train.py pad_full_data_autoencoder_model1.pt ignore.pt cuda:0 train_auc test_auc
# python tcr_ae_pep_lstm_train.py pad_full_data_autoencoder_model1.pt ignore_c.pt cuda:0 ae_c_train_auc ae_c_test_auc

if __name__ == '__main__':
    main(sys.argv)
