import os
import matplotlib.pyplot as plt

from functions import download_data, labelling
from functions.plot_utils import plot_images_mask_around_point

source = "PLEIADES"
dep = "MAYOTTE"
year = "2022"
task = "segmentation"
type_labeler = "BDTOPO"
tiles_size = "250"
n_bands = "3"
# Initialize S3 file system
fs = download_data.get_file_system()

print("\n*** 1- Téléchargement de la base d'annotation...\n")
labeler = labelling.get_labeler(type_labeler, year, dep, task)

os.makedirs(
    f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/",  # noqa
    exist_ok=True,
)

point = [-12.774895, 45.218719]
# point = [-12.783226, 45.219909]
# point = [-12.838770, 45.131530]
# point = [-12.961326, 45.126497]

plt.close()
images = plot_images_mask_around_point(point, source, dep, year, labeler, n_bands, fs, nb_dist=1)
images.savefig("data/images_point_mayotte2.png")
