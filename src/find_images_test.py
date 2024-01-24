import os
import matplotlib.pyplot as plt

from functions import download_data, labelling
from functions.plot_utils import plot_images_mask_around_point

# Mayotte
# dep = "MAYOTTE"
# point = [-12.838770, 45.131530]
# point = [-12.774895, 45.218719]
# point = [-12.719838, 45.117112]
point = [-12.961326, 45.126497]

# Martinique
dep = "MARTINIQUE"
# point = [14.650193, -61.055606]
point = [14.574659, -60.975153]

source = "PLEIADES"
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

os.makedirs(
    f"data/images-test-{dep.lower()}/",  # noqa
    exist_ok=True,
)

plt.close()
images = plot_images_mask_around_point(point, source, dep, year, labeler, n_bands, fs, nb_dist=1)
images.savefig(f"data/images-test-{dep.lower()}/images_{point}_{dep}_{year}.png")
