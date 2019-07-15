import torch
import torch.autograd as autograd
import csv
from old_configurations.ae_model import AutoencoderLSTMClassifier
import sys


def get_lists_from_pairs(pairs_file, max_len):
    tcrs = []
    peps = []
    with open(pairs_file, 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            tcr, pep = line
            if len(tcr) > max_len:
                continue
            tcrs.append(tcr)
            peps.append(pep)
    return tcrs, peps


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


def get_batches(tcrs, peps, tcr_atox, pep_atox, batch_size, max_length):
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
        padded_peps, pep_lens = pad_batch(batch_peps)
        batches.append((tcr_tensor, padded_peps, pep_lens))
        # Update index
        index += batch_size
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    padding_tcrs = ['X'] * missing
    padding_peps = ['A'] * missing
    convert_data(padding_tcrs, padding_peps, tcr_atox, pep_atox, max_length)
    batch_tcrs = tcrs[index:] + padding_tcrs
    tcr_tensor = torch.zeros((batch_size, max_length, 21))
    for i in range(batch_size):
        tcr_tensor[i] = batch_tcrs[i]
    batch_peps = peps[index:] + padding_peps
    padded_peps, pep_lens = pad_batch(batch_peps)
    batches.append((tcr_tensor, padded_peps, pep_lens))
    # Update index
    index += batch_size
    # Return list of all batches
    return batches


def predict(pairs_file, device, ae_file, model_file):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
    tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}

    # Set all parameters and program arguments
    device = device if torch.cuda.is_available() else 'cpu'
    args = {}
    args['ae_file'] = ae_file
    params = {}
    params['emb_dim'] = 10
    params['enc_dim'] = 30
    params['train_ae'] = True

    # Load autoencoder params
    checkpoint = torch.load(args['ae_file'])
    params['max_len'] = checkpoint['max_len']
    params['batch_size'] = checkpoint['batch_size']

    # test
    test_tcrs, test_peps = get_lists_from_pairs(pairs_file, params['max_len'])
    tcrs_copy = test_tcrs.copy()
    peps_copy = test_peps.copy()
    test_batches = get_batches(test_tcrs, test_peps, tcr_atox, pep_atox, params['batch_size'], params['max_len'])

    # load model
    model = AutoencoderLSTMClassifier(params['emb_dim'], device, params['max_len'], 21, params['enc_dim'],
                                      params['batch_size'], args['ae_file'], params['train_ae'])

    trained_model = torch.load(model_file)
    model.load_state_dict(trained_model['model_state_dict'])
    model.eval()
    model.to(device)
    results = []
    for batch in test_batches:
        tcrs, padded_peps, pep_lens = batch
        # Move to GPU
        tcrs = torch.tensor(tcrs).to(device)
        padded_peps = padded_peps.to(device)
        pep_lens = pep_lens.to(device)
        probs = model(tcrs, padded_peps, pep_lens)
        results.extend([prob.item() for prob in probs])
    results = results[:len(tcrs)]
    result_list = []
    for tcr, pep, prob in zip(tcrs_copy, peps_copy, results):
        result_list.append([tcr, pep, str(prob)])
    return result_list


if __name__ == '__main__':
    pairs = sys.argv[1]
    device = sys.argv[2]
    tcr_autoencoder = sys.argv[3]
    model = sys.argv[4]
    results = predict(pairs, device, tcr_autoencoder, model)
    print(results)
