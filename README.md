# ERGO
ERGO is a deep learing based model for predicting TCR-peptide binding.

required python packages:
```text
torch
numpy
sklearn
```

All runs use the main ERGO.py file.

### Model Training
For training the ERGO model, run:
```commandline
python ERGO.py model_type dataset sampling device 
```
When: `model_type` is `ae` for the TCR-autoencoder based model, or `lstm` for the lstm based model.
`dataset` is `mcpas` for McPAS-TCR dataset, or `vdjdb` for VDJdb dataset.
`sampling` can be `specific` for distinguishing different binders, `naive` for separating non-binders and binders,
or `memory` for distinguishing binders and memory TCRs.
`device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device.

Add the flag `--protein` suit the model for protein binding instead of specific peptide binding.
Add the argument `--train_auc_file=file` to write down the model train AUC during training.
Add the argument `--test_auc_file=file` to write down the model validation AUC during training.
Add the argument `--model_file=file.pt` in order to save the trained model.
Add the argument `--test_data_file=file.pickle` in order to save the test data.


#### TCR Autoencoder
The autoencoder based model requires a pre-trained TCR-autoencoder.
for training the TCR-autoencoder, go to the TCR-Autoencoder directory using
```commandline
cd TCR_Autoencoder
```
and run:
```commandline
python train_tcr_autoencoder.py BM_data_CDR3s device model_file.pt
```
when `device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device.
The trained autoencoder will be saved in `model_file` as a pytorch model.
You can use the already trained `tcr_autoencoder.pt` model instead.

When you run the ERGO.py file, use the argument `--ae_file=trained_autoencoder_file`

### Specific peptides or proteins binding evaluation
In order to evaluate the model for specific peptides or proteins,
first change the ERGO.py file last lines, such that the `main(args)` call will be in comment,
and the `pep_test(args)` or `protein_test(args) `  call will be active respectively.

Run:
```commandline
python ERGO.py model_type dataset sampling device --model_file=file.pt --test_data_file=file.pickle
```
When: `model_type` is `ae` for the TCR-autoencoder based model, or `lstm` for the lstm based model.
`dataset` is `mcpas` for McPAS-TCR dataset, or `vdjdb` for VDJdb dataset.
`sampling` can be `specific` for distinguishing different binders, `naive` for separating non-binders and binders,
or `memory` for distinguishing binders and memory TCRs.
`device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device.
(All arguments same as the model which is now evaluated)

The argument `--model_file=file.pt` is the trained model to be evaluated.
The argument `--test_data_file=file.pickle` is the test data (which the model has not seen during training,
in order to do so please save it in the training run command).

Add the flag `--protein` suit the model for protein binding instead of specific peptide binding.

## References
1. Springer, I., Besser, H., Tickotsky-Moskovitz, N., Dvorkin, S. & Louzoun, Y.
Prediction of specific TCR-peptide binding from large dictionaries of TCR-peptide pairs.
bioRxiv 650861 (2019). doi:10.1101/650861
