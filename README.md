# ERGO
ERGO is a deep learning based model for predicting TCR-peptide binding.

Check our web-tool at http://tcr.cs.biu.ac.il

## Requirements
```text
pytorch 1.4.0
numpy 1.18.1
scikit-learn 0.22.1
```

## Model Training
The main module for training is `ERGO.py`.
For training, run:
```commandline
python ERGO.py train model_type database specific gpu --model_file=model.pt --train_data_file=train_data --test_data_file=test_data
```
where:
- `model_type` is the the type of TCR encoding, LSTM based with `lstm` or autoencoder based with `ae`
- database is the training database, [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/) with `mcpas` 
or [VDJdb](https://vdjdb.cdr3.net/) with `vdjdb`.
- `gpu` is cuda device to use (e.g. `cuda:0`), or `cpu` for CPU (but it might be way slower)
- `--model_file` is the file which the model is saved to after training.
- `--train_data_file` and `--test_data_file` are train and test data files, you can set them as `auto` for defaults. 

## Binding Prediction
If you are interested in prediction only and not interested in training ERGO models,
It might be more convenient to use our web tool, available [here](http://tcr.cs.biu.ac.il).
You can choose what model and training set to use, and get the binding score of
given TCRs and peptides from a csv file.

Anyway you can also predict using the `ERGO.py` module.
It is quite similar to training, run:
```commandline
python ERGO.py predict model_type database specific gpu --model_file=model.pt --train_data_file=train_data --test_data_file=test_data
```
where:
- `--model_file` is the trained model file.
- `--test_data_file` is a csv file with TCR and peptide columns. See example file in the ERGO website.
- All other cmd parameters are similar to the training process. 

## Models
The trained models and some of the train/test datasets we used are stored in the models directory.

## TCR Autoencoder
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

## References
1. [Springer I, Besser H, Tickotsky-Moskovitz N, Dvorkin S and Louzoun Y (2020)
Prediction of Specific TCR-Peptide Binding From Large Dictionaries of TCR-Peptide Pairs.
Front. Immunol. 11:1803. doi: 10.3389/fimmu.2020.01803](https://www.frontiersin.org/articles/10.3389/fimmu.2020.01803/full)
