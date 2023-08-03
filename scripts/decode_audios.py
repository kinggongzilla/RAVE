import logging
import os

import gin
import numpy as np
from absl import flags, app
from tqdm import tqdm
import torch
import torchaudio

import rave
from rave.dataset import get_dataset, LatentDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "latents_path",
    default=None,
    required=True,
    help="path to generated latent tensors.",
)
flags.DEFINE_string(
    "decoded_output_path",
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

    #LOAD DATASET
    dataset = LatentDataset(
            FLAGS.latents_path,
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

    #INFERENCE: loop over batches and smaples data_loader and decode
    for i, batch in tqdm(enumerate(data_loader)):
        batch = batch.to(accelerator)
        decoded = model.decode(batch)

        #for each sample in batch save numpy array and audio file
        for j, sample in enumerate(decoded):
            #save as numpy array
            np.save(f'{FLAGS.decoded_output_path}/{i}_{j}.npy', sample.cpu().numpy())

            #save as audio file
            torchaudio.save(f'{FLAGS.decoded_output_path}/{i}_{j}.wav', sample.cpu(), model.sr)