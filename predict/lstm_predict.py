import torch
import sys
import numpy as np
import torch.autograd as autograd
from prediction.lstm_model import DoubleLSTMClassifier
import csv
import sys


def get_lists_from_pairs(pairs_file):
    tcrs = []
    peps = []
    with open(pairs_file, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            tcr, pep = line
            tcrs.append(tcr)
            peps.append(pep)
    return tcrs, peps


def convert_data(tcrs, peps, amino_to_ix):
    for i in range(len(tcrs)):
        if any(letter.islower() for letter in tcrs[i]):
            print(tcrs[i])
        tcrs[i] = [amino_to_ix[amino] for amino in tcrs[i]]
    for i in range(len(peps)):
        peps[i] = [amino_to_ix[amino] for amino in peps[i]]


def get_batches(tcrs, peps, batch_size, amino_to_ix):
    """
    Get batches from the data
    """
    # Initialization
    batches = []
    index = 0
    # Go over all data
    while index < len(tcrs):
        # Get batch sequences and math tags
        batch_tcrs = tcrs[index:index + batch_size]
        batch_peps = peps[index:index + batch_size]
        # Update index
        index += batch_size
        # Pad the batch sequences
        padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
        padded_peps, pep_lens = pad_batch(batch_peps)
        # Add batch to list
        batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens))
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    padding_tcrs = ['A'] * missing
    padding_peps = ['A'] * missing
    convert_data(padding_tcrs, padding_peps, amino_to_ix)
    batch_tcrs = tcrs[index:] + padding_tcrs
    batch_peps = peps[index:] + padding_peps
    padded_tcrs, tcr_lens = pad_batch(batch_tcrs)
    padded_peps, pep_lens = pad_batch(batch_peps)
    # Add batch to list
    batches.append((padded_tcrs, tcr_lens, padded_peps, pep_lens))
    # Update index
    index += batch_size
    # Return list of all batches
    return batches


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


def predict(pairs_file, device, model_file):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    # Set all parameters and program arguments
    device = device if torch.cuda.is_available() else 'cpu'
    params = {}
    params['batch_size'] = 50
    params['lstm_dim'] = 30
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0

    # test
    test_tcrs, test_peps = get_lists_from_pairs(pairs_file)
    tcrs_copy = test_tcrs.copy()
    peps_copy = test_peps.copy()
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, params['batch_size'], amino_to_ix)

    # load model
    model = DoubleLSTMClassifier(params['emb_dim'], params['lstm_dim'], params['dropout'], device)
    trained_model = torch.load(model_file)
    model.load_state_dict(trained_model['model_state_dict'])
    model.eval()
    model.to(device)
    results = []
    for batch in test_batches:
        padded_tcrs, tcr_lens, padded_peps, pep_lens = batch
        # Move to GPU
        padded_tcrs = padded_tcrs.to(device)
        tcr_lens = tcr_lens.to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(padded_tcrs, tcr_lens, padded_peps, pep_lens)
        results.extend([prob.item() for prob in probs])
    results = results[:len(test_tcrs)]
    result_list = []
    for tcr, pep, prob in zip(tcrs_copy, peps_copy, results):
        result_list.append([tcr, pep, str(prob)])
    return result_list


def main(pairs_file):
    results = predict(pairs_file, 'cuda:0', 'predict/lstm_model.pt')
    return results


if __name__ == '__main__':
    results = predict(sys.argv[1], 'cuda:0', 'lstm_model.pt')
    print(results)
