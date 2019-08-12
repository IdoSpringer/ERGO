# ERGO
ERGO is a deep learing based model for predicting TCR-peptide binding.

### Python Dependencies and Requirements:
```text
pytorch 1.0.0
numpy 1.15.4
scikit-learn 0.19.2
```
- The code has been tested on Linux 3.10.0-957.27.2.el7.x86_64.
- The code runs faster with GPU usage, but it should work also on CPU. It was developed on GPU
(Nvidia Tesla K40m with CUDA 10.0).

For initialization, download this repository or clone using
`git clone https://github.com/IdoSpringer/ERGO`. It should take a few seconds.

### Model Training
Training a model should take about an hour.
All runs use the main ERGO.py file.

For training the ERGO model, run:
```commandline
python ERGO.py train model_type dataset sampling device 
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

When you run the ERGO.py file, use the argument `--ae_file=trained_autoencoder_file`.
You can use the already trained `tcr_autoencoder.pt` model instead, with `--ae_file=auto`.

### Specific peptides or proteins binding evaluation
In order to evaluate the model for specific peptides or proteins,
run:
```commandline
python ERGO.py test model_type dataset sampling device --model_file=file.pt --test_data_file=file.pickle
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

### Prediction
You can use the models you have trained to predict new data,
or you can use our already trained models in the models directory.

Data for prediction must be in a .csv format,
where the first column is the TCRs and the second column is the peptides.
See [example](pairs_example.csv).

For binding prediction, run:
```commandline
python ERGO.py predict model_type dataset sampling device --model_file=file.pt --test_data_file=file.csv
```
When: `model_type` is `ae` for the TCR-autoencoder based model, or `lstm` for the lstm based model.
`dataset` is `mcpas` for McPAS-TCR dataset, or `vdjdb` for VDJdb dataset.
`sampling` can be `specific` for distinguishing different binders, `naive` for separating non-binders and binders,
or `memory` for distinguishing binders and memory TCRs.
`device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device.

The argument `--model_file=file.pt` is the trained model.
If you want to use our trained models, use `--model_file=auto`, and the code will choose the right trained
model according to the arguments `model_type`, `dataset` and `sampling`.
If you are using your model file, make sure that those arguments match the model.

The argument `--test_data_file=file.csv` is the test data to predict in .csv format.
If you want to run our example use `--test_data_file=auto`.

The code should print the input TCRs and the peptides, with the predicted binding probabilities.
Notice that in the autoencoder model, TCR max length is 28, so longer sequences will be ignored.
Prediction should be take a few seconds.

## References
1. Springer, I., Besser, H., Tickotsky-Moskovitz, N., Dvorkin, S. & Louzoun, Y.
Prediction of specific TCR-peptide binding from large dictionaries of TCR-peptide pairs.
bioRxiv 650861 (2019). doi:10.1101/650861
