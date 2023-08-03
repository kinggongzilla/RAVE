import logging
import os

import gin
import numpy as np
from absl import flags
from tqdm import tqdm
import torch

import rave
from torch.utils.data import DataLoader
from rave.dataset import AudioDataset


logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "db_path",
    default=None,
    required=True,
    help="path to database.",
)
flags.DEFINE_string(
    "encoded_output_path",
    default=None,
    required=True,
    help="path to encoded samples.",
)
flags.DEFINE_string('run',
                    default=None,
                    help='Path to previous run folder')
flags.DEFINE_integer('workers',
                     default=10,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_integer('batch', 8, help='Batch size')


def main(argv):
    accelerator = "cuda"

    # model = rave.RAVE()

    #LOAD DATASET
    dataset = AudioDataset(
            FLAGS.db_path,
        )
    num_workers = FLAGS.workers

    data_loader = DataLoader(dataset,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)



    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)

    print(f"using {checkpoint}")

    model = rave.RAVE()
    model.load_state_dict(torch.load(checkpoint)["state_dict"])
    model.to(accelerator)
    model.eval()

    #INFERENCE: loop over data_loader and encode
    for i, batch in tqdm(enumerate(data_loader)):
        batch = batch.to(accelerator)
        encoded = model.encode(batch)

        #for each sample in batch
        for j in range(FLAGS.batch):
            #SAVE ENCODED DATA
            np.save(f'{FLAGS.encoded_output_path}/{i}_{j}.npy', encoded[j].cpu().numpy())

        # #SAVE ENCODED DATA
        # np.save(f'{FLAGS.encoded_output_path}/{i}.npy', encoded.cpu().numpy())