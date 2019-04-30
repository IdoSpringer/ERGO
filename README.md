# ERGO
ERGO is a deep learing based model for predicting TCR-peptide binding.

required python packages:
```text
torch
numpy
sklearn
```

### Model Training

#### LSTM Based Model
The LSTM based model is built in the `lstm_model.py` file.
For training the model, run:
```commandline
python lstm_train.py model_file.pt device data_path
```
when `device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device,
and `data_path` is the path for the train + validation data (one file from data directory).
The trained autoencoder will be saved in `model_file` as a pytorch model.
You can use the already trained `lstm_model.pt` model instead.

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

#### Autoencoder Based Model
The LSTM based model is built in the `ae_model.py` file.
For training the model, run:
```commandline
python ae_train.py model_file.pt device tcr_autoencoder_path data_path
```
when `device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device,
`tcr_autoencoder_path` is the trained tcr_autoencoder model file path,
and `data_path` is the path for the train + validation data (one file from data directory).
The trained autoencoder will be saved in `model_file` as a pytorch model.
You can use the already trained `ae_model.pt` model instead.

### Prediction
For prediction, we recommend using the [ERGO Website]() application.
You can also use the relevant files in this repository.
Data for prediction should be in .csv format as described in the website.
See [this file]() for example.

#### LSTM Based Model
For prediction, run:
```commandline
python lstm_predict.py pairs_file device lstm_based_model_path
```
when `pairs_file` is the data for prediction, 
`device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device,
and `lstm_based_model_path` is the trained tcr_autoencoder model file path.
You can use the already trained models, by running
```commandline
python lstm_predict.py pairs_file device lstm_model.pt
```
#### Autoencoder Based Model
For prediction, run:
```commandline
python ae_predict.py pairs_file device tcr_autoencoder_path autoencoder_based_model_path
```
when `pairs_file` is the data for prediction, 
`device` is a CUDA GPU device (e.g. 'cuda:0') or 'cpu' for CPU device,
`tcr_autoencoder_path` is the trained tcr_autoencoder model file path,
and `autoencoder_based_model_path` is the trained autoencoder based model file path.
You can use the already trained models, by running
```commandline
python ae_predict.py pairs_file device TCR_Autoencoder/tcr_autoencoder.pt ae_model.pt
```

### References
[1] (...), High precision specific TCR-peptide binding prediction for repertoire based biomarkers, (...) .