import base64
import logging
import os
import sys

import flask
import numpy as np
from absl import flags
from udls import AudioExample

import rave
from rave.dataset import get_dataset
from torch.utils.data import DataLoader


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
    "encoded_data_path",
    default=None,
    required=True,
    help="path to encoded samples.",
)
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_integer('n_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_integer('batch', 8, help='Batch size')


def main(argv):
    accelerator = "cuda"

    model = rave.RAVE()
    model.eval()

    #LOAD DATASET
    dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                       model.sr,
                                       FLAGS.n_signal,
                                       derivative=False, #only set to true if you trained on derivative
                                       normalize=False) #only set to true if you trained on normalized
    num_workers = FLAGS.workers

    data_loader = DataLoader(dataset,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)


    #LOAD MODEL: load from checkpoint
    run_path = rave.core.search_for_run(FLAGS.ckpt) #returns path to .ckpt file

    model = rave.RAVE.load_from_checkpoint(run_path, map_location=accelerator) 

    #INFERENCE: loop over data_loader and encode
    for batch in data_loader:
        batch = batch.to(accelerator)
        encoded = model.encode(batch)

        #SAVE ENCODED DATA
        np.save(f'FLAGS.encoded_data_path', encoded.cpu().numpy())

