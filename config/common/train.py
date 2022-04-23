from simrec.config import LazyCall

train = dict(
    # Directory where output files are written
    output_dir = "./output",

    train_batch_size = 4,

    epochs = 25,

    distributed = dict(
        node_id = 0,
        world_size = 1,
        dist_utl = "tcp://127.0.0.1:12345",
        multiprocessing_distributed = True,
        rank = 0
    ),

    multi_scale = [[224,224],[256,256],[288,288],[320,320],[352,352],
                   [384,384],[416,416],[448,448],[480,480],[512,512],
                   [544,544],[576,576],[608,608]]
)