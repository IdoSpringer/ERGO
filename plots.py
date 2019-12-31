import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import evaluation_methods as eval
import argparse
import os
from scipy import stats


# Table 1 - SPB
def spb_auc(args, peps):
    dir = 'final_results'
    aucs = np.zeros((len(peps), 5))
    for i in range(1, 5 + 1):
        args.model_file = dir + '/' + '_'.join([args.model_type, args.dataset + str(i) + '.pt'])
        args.train_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, 'train' + str(i) + '.pickle'])
        args.test_data_file = dir + '/' + '_'.join([args.model_type, args.dataset, 'test' + str(i) + '.pickle'])
        model, data = eval.load_model_and_data(args)
        train_data, test_data = data
        for k, pep in enumerate(peps):
            try:
                auc = eval.single_peptide_score(args, model, test_data, pep, None)[0]
                # print(pep, auc)
                aucs[k, i - 1] = auc
            except ValueError:
                print('None')
            except IndexError:
                print('None')
    # print(aucs)
    aucs = ma.array(aucs, mask=aucs == 0)
    # print(aucs)
    print(ma.mean(aucs, axis=1))


# Figure 1
# A. LSTM drawing (not in code)
# B. AE drawing (not in code)
# C. Accuracy as num of classes
# D. SPB ROC curve
# E. TPP ROC curve

# todo nice and coherent colors


def acc_plot():
    # plot errorbar of accuracy per number of classes
    # currently we only take reported results
    ae_mcpas_accs = [[0.69, 0.589, 0.52, 0.521],
                     [0.708, 0.572, 0.502, 0.508],
                     [0.724, 0.599, 0.501, 0.454],
                     [0.717, 0.601, 0.538, 0.472],
                     [0.718, 0.582, 0.523, 0.534]]
    lstm_mcpas_accs = [[0.688, 0.559, 0.503, 0.432],
                       [0.645, 0.501, 0.464, 0.41],
                       [0.657, 0.522, 0.46, 0.41],
                       [0.683, 0.534, 0.493, 0.446],
                       [0.671, 0.549, 0.493, 0.409]]
    ae_vdjdb_accs = [[0.702, 0.628, 0.599, 0.572],
                     [0.699, 0.622, 0.593, 0.568],
                     [0.695, 0.618, 0.589, 0.564],
                     [0.699, 0.632, 0.6, 0.571]]
    lstm_vdjdb_accs = [[0.624, 0.541, 0.503, 0.453],
                       [0.631, 0.546, 0.492, 0.434],
                       [0.638, 0.513, 0.473, 0.436],
                       [0.622, 0.532, 0.496, 0.458],
                       [0.638, 0.543, 0.493, 0.449]]
    ae_mcpas_means = np.average(ae_mcpas_accs, axis=0)
    ae_mcpas_std = np.std(ae_mcpas_accs, axis=0)
    ae_vdjdb_means = np.average(ae_vdjdb_accs, axis=0)
    ae_vdjdb_std = np.std(ae_vdjdb_accs, axis=0)
    lstm_mcpas_means = np.average(lstm_mcpas_accs, axis=0)
    lstm_mcpas_std = np.std(lstm_mcpas_accs, axis=0)
    lstm_vdjdb_means = np.average(lstm_vdjdb_accs, axis=0)
    lstm_vdjdb_std = np.std(lstm_vdjdb_accs, axis=0)
    print(ae_mcpas_means, ae_mcpas_std)
    print(ae_vdjdb_means, ae_vdjdb_std)
    print(lstm_mcpas_means, lstm_mcpas_std)
    print(lstm_vdjdb_means, lstm_vdjdb_std)
    classes = range(2, 5 + 1)
    plt.errorbar(classes, ae_mcpas_means, yerr=ae_mcpas_std, label='AE, McPAS', color='royalblue', linestyle='-')
    plt.errorbar(classes, ae_vdjdb_means, yerr=ae_vdjdb_std, label='AE, VDJdb', color='tomato', linestyle='-')
    plt.errorbar(classes, lstm_mcpas_means, yerr=lstm_mcpas_std, label='LSTM, McPAS', color='royalblue', linestyle='--')
    plt.errorbar(classes, lstm_vdjdb_means, yerr=lstm_vdjdb_std, label='LSTM, VDJdb', color='tomato', linestyle='--')
    plt.legend()
    plt.title('MPS Accuracy per number of classes')
    plt.xlabel('Number of classes')
    plt.xticks(classes)
    plt.ylabel('Mean accuracy')
    plt.show()


def spb_roc(args, peptides):
    # single peptides roc
    model, data = eval.load_model_and_data(args)
    train_data, test_data = data
    colors = ['indigo', 'darkolivegreen', 'darkcyan']
    for pep, color in zip(peptides, colors):
        fpr, tpr, thresholds = eval.single_peptide_score(args, model, test_data, pep, None)[1]
        plt.plot(fpr, tpr, label=pep, color=color)
    plt.title('SPB ROC Curve, AE McPAS Model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()


def ttp_roc(args):
    # roc for tpp-i, tpp-ii, tpp-iii
    model, data = eval.load_model_and_data(args)
    train_data, test_data = data
    new_test_tcrs, new_test_peps = eval.extract_new_tcrs_and_peps(train_data, test_data)
    fpr, tpr, thresholds = eval.new_pairs_score(args, model, test_data)[1]
    plt.plot(fpr, tpr, label='TPP-I', color='orange')
    fpr, tpr, thresholds = eval.new_tcrs_score(args, model, test_data, new_test_tcrs)[1]
    plt.plot(fpr, tpr, label='TPP-II', color='springgreen')
    fpr, tpr, thresholds = eval.new_peps_score(args, model, test_data, new_test_tcrs, new_test_peps)[1]
    plt.plot(fpr, tpr, label='TPP-III', color='dodgerblue')
    plt.title('TPP ROC Curve, AE McPAS Model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()


# Figure 2
# A. TPP AUC per number of pairs
# B. TPP AUC per missing amino-acids
# C. Number of TCRs per peptide statistics in McPAS/VDJdb datasets
# D. TPP AUC per number of TCRs per peptide

def max_auc(auc_file):
    with open(auc_file, 'r') as file:
        aucs = []
        for line in file:
            aucs.append(float(line.strip()))
        max_auc = max(aucs)
    return max_auc


def position_auc():
    dir = 'mis_pos/mis_pos_auc'
    mkeys = {'ae': 0, 'lstm': 1}
    dkeys = {'mcpas': 0, 'vdjdb': 1}
    directory = os.fsencode(dir)
    aucs = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        name = filename.split('_')
        mkey = name[0]
        dkey = name[1]
        mis = int(name[-1])
        iter = int(name[-2])
        #if mkey == 'lstm' and dkey == 'vdjdb' and mis > 27:
        #    continue
        auc = max_auc(dir + '/' + filename)
        aucs.append((mkeys[mkey], dkeys[dkey], iter, mis, auc))
    # we do this because the runs might not be done yet
    max_index0 = max(t[3] for t in aucs if t[0] == 0)
    max_index10 = max(t[3] for t in aucs if t[0] == 1 and t[1] == 0)
    # max_index11 = max(t[3] for t in aucs if t[0] == 1 and t[1] == 1)
    #max_index11 = 27
    max_index = max(max_index0, max_index10, max_index11)
    max_iter0 = max(t[2] for t in aucs if t[0] == 0)
    max_iter10 = max(t[2] for t in aucs if t[0] == 1 and t[1] == 0)
    max_iter11 = max(t[2] for t in aucs if t[0] == 1 and t[1] == 1)
    max_iter = max(max_iter0, max_iter10, max_iter11)
    auc_tensor = np.zeros((2, 2, max_iter + 1, max_index + 1))
    for auc in aucs:
        auc_tensor[auc[0], auc[1], auc[2], auc[3]] = auc[4]
    auc_tensor0 = auc_tensor[0, :, :max_iter0 + 1, :max_index0 + 1]
    # print('auc tensor 0')
    # print(auc_tensor0)
    auc_tensor10 = auc_tensor[1, 0, :max_iter10 + 1, :max_index10 + 1]
    # print('auc tensor 10')
    # print(auc_tensor10)
    auc_tensor11 = auc_tensor[1, 1, :max_iter11 + 1, :max_index11 + 1]
    # print('auc tensor 11')
    # print(auc_tensor11)
    means0 = np.mean(auc_tensor0, axis=1)
    std0 = stats.sem(auc_tensor0, axis=1)
    means10 = np.mean(auc_tensor10, axis=0)
    std10 = stats.sem(auc_tensor10, axis=0)
    means11 = np.mean(auc_tensor11, axis=0)
    std11 = stats.sem(auc_tensor11, axis=0)
    auc_means = [means0[0], means0[1], means10, means11]
    # print(auc_means)
    auc_stds = [std0[0], std0[1], std10, std11]
    labels = ['MsPAS, AE model', 'VDJdb, AE model', 'MsPAS, LSTM model', 'VDJdb, LSTM model']
    colors = ['dodgerblue', 'springgreen', 'dodgerblue', 'springgreen']
    styles = ['-', '-', '--', '--']
    for auc_mean, auc_std, label, color, style in zip(auc_means, auc_stds, labels, colors, styles):
        plt.errorbar(range(1, len(auc_mean) + 1), auc_mean, yerr=auc_std, label=label,
                    color=color, linestyle=style)
    plt.legend(loc=4, prop={'size': 8})
    #plt.set_xlabel('Missing amino acid index', fontdict={'fontsize': 11}, labelpad=1)
    #ax.set_ylabel('Best AUC score', fontdict={'fontsize': 14})
    #ax.set_title(title, fontdict={'fontsize': 16})
    plt.show()


def mis_pos():
    dir = 'mis_pos/mis_pos_auc'
    mkeys = {'ae': 0, 'lstm': 1}
    dkeys = {'mcpas': 0, 'vdjdb': 1}
    directory = os.fsencode(dir)
    aucs = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        name = filename.split('_')
        mkey = name[0]
        dkey = name[1]
        mis = int(name[-1])
        iter = int(name[-2]) - 1
        auc = max_auc(dir + '/' + filename)
        aucs.append((mkeys[mkey], dkeys[dkey], iter, mis, auc))
    max_index = 27
    max_iter = 5
    auc_tensor = np.zeros((2, 2, max_iter, max_index + 1))
    for auc in aucs:
        auc_tensor[auc[0], auc[1], auc[2], auc[3]] = auc[4]
    auc_tensor = ma.array(auc_tensor, mask=auc_tensor == 0)
    auc_mean = ma.mean(auc_tensor, axis=2)
    auc_std = ma.std(auc_tensor, axis=2)
    auc_means = auc_mean.reshape((2 * 2, -1))
    auc_stds = auc_std.reshape((2 * 2, -1))
    print(auc_means)
    print(auc_stds)

    labels = ['MsPAS, AE model', 'VDJdb, AE model', 'MsPAS, LSTM model', 'VDJdb, LSTM model']
    colors = ['dodgerblue', 'springgreen', 'dodgerblue', 'springgreen']
    styles = ['-', '-', '--', '--']
    for auc_mean, auc_std, label, color, style in zip(auc_means, auc_stds, labels, colors, styles):
        plt.errorbar(range(1, len(auc_mean) + 1), auc_mean, yerr=auc_std, label=label,
                     color=color, linestyle=style)
    plt.legend(loc=4, prop={'size': 8})
    plt.xlabel('Missing amino acid index')
    plt.ylabel('Mean AUC score')
    plt.title('TPP AUC per missing amino-acid')
    plt.show()

# copy some older code


if __name__ == '__main__':
    dir = 'final_results'
    train_data_file = dir + '/' + 'ae_mcpas_train1.pickle'
    test_data_file = dir + '/' + 'ae_mcpas_test1.pickle'
    model_file = dir + '/' + 'ae_mcpas1.pt'
    parser = argparse.ArgumentParser()
    parser.add_argument("function")
    parser.add_argument("--model_type", default='ae')
    parser.add_argument("--dataset", default='mcpas')
    parser.add_argument("--sampling", default='specific')
    parser.add_argument("--device", default='cuda:1')
    parser.add_argument("--protein", default="store_true")
    parser.add_argument("--hla", action="store_true")
    parser.add_argument("--ae_file")
    parser.add_argument("--train_auc_file")
    parser.add_argument("--test_auc_file")
    parser.add_argument("--model_file", default=model_file)
    parser.add_argument("--roc_file")
    parser.add_argument("--train_data_file", default=train_data_file)
    parser.add_argument("--test_data_file", default=test_data_file)
    args = parser.parse_args()

    if args.function == 'acc':
        acc_plot()
    elif args.function == 'spb_roc':
        peptides = ['GLCTLVAML', 'NLVPMVATV', 'GILGFVFTL']
        spb_roc(args, peptides)
    elif args.function == 'spb_auc':
        peptides = ['GLCTLVAML', 'NLVPMVATV', 'GILGFVFTL']
        spb_auc(args, peptides)
        vdjdb_peps = ["IPSINVHHY", "TPRVTGGGAM", "NLVPMVATV", "GLCTLVAML", "RAKFKQLL",
                      "YVLDHLIVV", "GILGFVFTL", "PKYVKQNTLKLAT", "CINGVCWTV", "KLVALGINAV",
                      "ATDALMTGY", "RPRGEVRFL", "LLWNGPMAV", "GTSGSPIVNR", "GTSGSPIINR",
                      "KAFSPEVIPMF", "TPQDLNTML", "EIYKRWII", "KRWIILGLNK", "FRDYVDRFYKTLRAEQASQE",
                      "GPGHKARVL", "FLKEKGGL"]
        if args.dataset == 'vdjdb':
            spb_auc(args, vdjdb_peps)
    elif args.function == 'ttp':
        ttp_roc(args)
    elif args.function == 'mis_pos':
        # position_auc()
        mis_pos()
        pass
