## Getting Started
Here we provide the basic tutorials about the usage SimREC. Make sure you've already install the environments for SimREC, please refer to [Installation]().

### Training
In SimREC, we provide `tools/train_engine.py` and `tools/eval_engine.py` for training and evaluation.

The following script will start training `simrec` model on `refcoco` dataset on a single GPU:
```shell
$ bash tools/train.sh configs/simrec_refcoco_scratch.py 1
```

All of the `checkpoints`, `logs` and `tensorboard` logs will be saved to `cfg.train.output_dir`, you can modify them in the config file:
```python
# config.py
from .common.train import train

train.output_dir = "/your/own/path"
```

For **distributed data parallel training**, you can simply modify the training script as follows:
```shell
$ bash tools/train.sh configs/simrec_refcoco_scratch.py 4
```
you can simple running ddp training by modifing the last number of the training scripts to `4`. 

### Override the config in command line
You can override the config file in command line. For example, you can enable the `SyncBatchNorm` in scripts like:
```shell
$ bash tools/train.sh configs/simrec_refcoco_scratch.py 4 train.sync_bn.enabled=True
```
which may give you a better result.