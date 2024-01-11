import os

import matplotlib.pyplot as plt

from functions import download_data, labelling
from functions.plot_utils import plot_square_nb_images_mask
from functions.image_utils import find_image_of_point, point_is_in_image

# from functions.image_utils import *
from astrovision.data import SatelliteImage

source = "PLEIADES"
dep = "MAYOTTE"
year = "2020"
task = "segmentation"
type_labeler = "BDTOPO"
tiles_size = "250"
# Initialize S3 file system
fs = download_data.get_file_system()

point = [-12.47202, 45.13075]
# find_image_of_point(point, dep, year, fs)
filepath = find_image_of_point([14.635338, -61.038345], "MARTINIQUE", "2022", fs)
image = SatelliteImage.from_raster(
    file_path=f"/vsis3/{filepath}",
    n_bands=3,
)
point_is_in_image(image, [14.635338, -61.038345])
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
