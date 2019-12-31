from ae_utils import *


# tcrs must have 21 one-hot, not 22. padding index in pep must be 0.
def convert_data(tcrs, peps, tcr_atox, pep_atox, max_len, mis):
    for i in range(len(tcrs)):
        tcrs[i] = pad_tcr(tcrs[i], tcr_atox, max_len, mis)
    convert_peps(peps, pep_atox)


def pad_tcr(tcr, amino_to_ix, max_length, mis):
    padding = torch.zeros(max_length, 20 + 1)
    tcr = tcr + 'X'
    for i in range(len(tcr)):
        if i == mis:
            continue
        amino = tcr[i]
        padding[i][amino_to_ix[amino]] = 1
    return padding


def get_batches(tcrs, peps, signs, tcr_atox, pep_atox, batch_size, max_length, mis):
    """
    Get batches from the data
    """
    # Initialization
    batches = []
    index = 0
    convert_data(tcrs, peps, tcr_atox, pep_atox, max_length, mis)
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


def get_full_batches(tcrs, peps, signs, tcr_atox, pep_atox, batch_size, max_length, mis):
    """
    Get batches from the data, including last with padding
    """
    # Initialization
    batches = []
    index = 0
    convert_data(tcrs, peps, tcr_atox, pep_atox, max_length, mis)
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
    # pad data in last batch
    missing = batch_size - len(tcrs) + index
    if missing < batch_size:
        padding_tcrs = ['X'] * missing
        padding_peps = ['A' * (batch_size - missing)] * missing
        convert_data(padding_tcrs, padding_peps, tcr_atox, pep_atox, max_length, mis)
        batch_tcrs = tcrs[index:] + padding_tcrs
        tcr_tensor = torch.zeros((batch_size, max_length, 21))
        for i in range(batch_size):
            tcr_tensor[i] = batch_tcrs[i]
        batch_peps = peps[index:] + padding_peps
        padded_peps, pep_lens = pad_batch(batch_peps)
        batch_signs = [0.0] * batch_size
        batches.append((tcr_tensor, padded_peps, pep_lens, batch_signs))
        # Update index
        index += batch_size
    # Return list of all batches
    return batches
