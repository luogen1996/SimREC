import torch.nn as nn

from simrec.config import LazyCall
from simrec.models.simrec import SimREC
from simrec.models.backbones.cspdarknet import CspDarkNet
from simrec.models.heads.rec_heads import REChead
from simrec.models.language_encoders.lstm_sa import LSTM_SA
from simrec.layers.fusion_layer import SimpleFusion, MultiScaleFusion, GaranAttention

model = LazyCall(SimREC)(
    visual_backbone=LazyCall(CspDarkNet)(
        pretrained=False,
        pretrained_weight_path="./data/weights/cspdarknet_coco.pth",
        freeze_backbone=True,
        multi_scale_outputs=True,
    ),
    language_encoder=LazyCall(LSTM_SA)(
        depth=1,
        hidden_size=512,
        num_heads=8,
        ffn_size=2048,
        flat_glimpses=1,
        word_embed_size=300,
        dropout_rate=0.,
        # language_encoder.pretrained_emb and language.token_size is meant to be set
        # before instantiating
        freeze_embedding=True,
        use_glove=True,
    ),
    multi_scale_manner=LazyCall(MultiScaleFusion)(
        v_planes=(512, 512, 512),
        scaled=True
    ),
    fusion_manner=LazyCall(nn.ModuleList)(
        modules = [
            LazyCall(SimpleFusion)(v_planes=256, out_planes=512, q_planes=512),
            LazyCall(SimpleFusion)(v_planes=512, out_planes=512, q_planes=512),
            LazyCall(SimpleFusion)(v_planes=1024, out_planes=512, q_planes=512),
        ]
    ),
    attention_manner=LazyCall(GaranAttention)(
        d_q=512,
        d_v=512
    ),
    head=LazyCall(REChead)(
        label_smooth=0.,
        num_classes=0,
        width=1.0,
        strides=[32,],
        in_channels=[512,],
        act="silu",
        depthwise=False
    )
)