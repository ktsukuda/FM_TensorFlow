# FM_TensorFlow

Factorization Machine with TensorFlow.

## Environment

- Python: 3.6
- TensorFlow: 2.2.0
- CUDA: 10.1
- Ubuntu: 18.04

## Dataset

The MovieLens dataset is used. You can download the data for training/validation/test from [He Xiangnan's repositroy](https://github.com/hexiangnan/neural_factorization_machine). The downloaded data (ml-tag.train.libfm, ml-tag.validation.libfm, and ml-tag.test.libfm) should be put in `data/ml-tag`.

## Run the Codes

```bash
$ python FM_TensorFlow/main.py
```

## Details

The hyperparameters (batch_size, lr, l2_weight, and latent_dim) are tuned by using the valudation data in terms of RMSE. See [config.ini](https://github.com/ktsukuda/FM_TensorFlow/blob/master/FM_TensorFlow/config.ini) about the range of each hyperparameter.

By running the code, hyperparameters are automatically tuned. After the training process, the best hyperparameters and RMSE computed by using the test data are displayed.

Given a specific combination of hyperparameters, the corresponding training results are saved in `data/train_result/<hyperparameter combination>` (e.g., data/train_result/batch_size_1024-lr_0.01-l2_weight_0.001-latent_dim_64-epoch_3). In the directory, model files and a json file (`epoch_data.json`) that describes information for each epoch are generated. The json file can be described as follows (epoch=3).

```json
[
    {
        "epoch": 0,
        "loss": 372591.2521972656,
        "RMSE": 0.6144263698333835
    },
    {
        "epoch": 1,
        "loss": 230172.80307006836,
        "RMSE": 0.5803952708097087
    },
    {
        "epoch": 2,
        "loss": 203277.17392730713,
        "RMSE": 0.5679699090405614
    }
]
```
