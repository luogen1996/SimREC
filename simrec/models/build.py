from .mcn import MCN
from .simrec import SimREC

def build_model(config, pretrained_emb, token_size):
    model_use = config.MODEL
    if model_use == "simrec":
        return SimREC(config, pretrained_emb, token_size)
    elif model_use == "mcn":
        return MCN(config, pretrained_emb, token_size)
