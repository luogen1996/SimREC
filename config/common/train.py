from simrec.config import LazyCall

train = dict(
    # Directory where output files are written
    output_dir = "./output",

    train_batch_size = 4,

    epochs = 25,

    dist = dict(
        node_id = 0,
        world_size = 1,
        dist_utl = "tcp://127.0.0.1:12345",
        multiprocessing_distributed = True,
        rank = 0
    )
)