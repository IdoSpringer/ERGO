import pickle


def extract_pep_tcr_files(pep):
    dir = 'comp_logos'
    train_data_file = '_'.join(['mcpas', 'train.pickle'])
    test_data_file = '_'.join(['mcpas', 'test.pickle'])
    # Read train data
    with open(train_data_file, "rb") as file:
        train = pickle.load(file)
    # Read test data
    with open(test_data_file, "rb") as file:
        test = pickle.load(file)
    count = 0
    with open('_'.join([pep, 'pos']), 'w') as pos:
        for (tcr, p, sign) in train + test:
            if p[0] == pep and sign == 'p' and len(tcr) == 13:
                pos.write(tcr + '\n')
                count += 1
    with open('_'.join([pep, 'neg']), 'w') as neg:
        for (tcr, p, sign) in train + test:
            if p[0] == pep and sign == 'n' and count and len(tcr) == 13:
                neg.write(tcr + '\n')
                count -= 1


peptides = ['GLCTLVAML', 'NLVPMVATV', 'GILGFVFTL']
for pep in peptides:
    extract_pep_tcr_files(pep)
