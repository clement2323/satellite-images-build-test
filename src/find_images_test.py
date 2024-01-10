import os

import matplotlib.pyplot as plt

from functions import download_data, labelling
from functions.plot_utils import plot_square_nb_images_mask

source = "PLEIADES"
dep = "MAYOTTE"
year = "2022"
task = "segmentation"
type_labeler = "BDTOPO"
tiles_size = "250"
# Initialize S3 file system
fs = download_data.get_file_system()

print("\n*** 1- Téléchargement de la base d'annotation...\n")
labeler = labelling.get_labeler(type_labeler, year, dep, task)

os.makedirs(
    f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/",  # noqa
    exist_ok=True,
)

n_bands = "3"
debut = 200
fin = 209
plt.close()
images = plot_square_nb_images_mask(source, dep, year, labeler, debut, fin, n_bands, fs)
images.savefig("data/coucou.png")
