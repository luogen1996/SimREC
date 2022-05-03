## Getting Started
Here we provide the basic tutorials about the usage SimREC. Make sure you've already install the environments for SimREC, please refer to [Installation]().

### Training
In SimREC, we provide `tools/train_engine.py` and `tools/eval_engine.py` for training and evaluation.

The following script will start training `simrec` model on `refcoco` dataset on a single GPU:
```shell
$ bash tools/train.sh configs/simrec_refcoco_scratch.py 1
```

All of the `checkpoints`, `logs` and `tensorboard` logs will be saved to `cfg.train.output_dir`, you can modify them in the config file, we highly recommend the users to put their own config file under `/configs` to easily reuse the default config files:
```python
# config.py
from .common.train import train

train.output_dir = "/your/own/path"
```

For **distributed data parallel training**, you can modify the training script as follows:
```shell
$ bash tools/train.sh configs/simrec_refcoco_scratch.py 4
```
To run ddp training mode by simply modifing the last number of the training scripts to `4`. 

### Override the config in command line
You can override the config file in command line. For example, you can enable the `SyncBatchNorm` in scripts for ddp training like:
```shell
$ bash tools/train.sh configs/simrec_refcoco_scratch.py 4 train.sync_bn.enabled=True
```
which may give you a better result.


### Resume training
In SimREC, we support two resume training ways adopted from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer):

- Automatically resume training:

SimREC automatically saves `last_checkpoint.pth` during training time to `cfg.train.output_dir`, if set `cfg.train.auto_resume.enabled=True`, before training, SimREC will find if there is `last_checkpoint.pth` in `cfg.train.output_dir` and automatically resume from it.

- Resume training from specific checkpoint:

Firstly, you should disable `auto-resume` function which will override the `cfg.train.resume_path` by setting `cfg.train.auto_resume.enabled=False`, and you should update `cfg.train.resume_path` to the specific checkpoint you want to resume from as follows:
```python
# config.py
from .common.train import train

train.auto_resume.enabled=False
train.resume_path = "path/to/specific/checkpoint.pth"
```

### Evaluation

Run `bash tools/eval.sh` under to evaluate the saved checkpoint.

```shell
bash tools/eval.sh config/simrec_refcoco_scratch.py 4 /path/to/checkpoint
```
