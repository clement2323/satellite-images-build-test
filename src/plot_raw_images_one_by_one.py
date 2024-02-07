import os
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
from functions import download_data, labelling
import numpy as np
import matplotlib.pyplot as plt


dep = "MAYOTTE"
source = "PLEIADES"
year = "2022"
task = "segmentation"
type_labeler = "BDTOPO"
tiles_size = "2000"
n_bands = "3"
# Initialize S3 file system
fs = download_data.get_file_system()

image_filepaths = fs.ls(f"projet-slums-detection/data-raw/{source}/{dep}/{year}")

# Affichage des résultats

os.makedirs(
    f"../data/affichage_data_raw/{type_labeler}/{task}/{source}/{dep}/{year}/",  # noqa
    exist_ok=True,
)

labeler = labelling.get_labeler(type_labeler, year, dep, task)

for im in list(image_filepaths):
    # 1- Ouvrir avec SatelliteImage
    image = SatelliteImage.from_raster(
        file_path=f"/vsis3/{im}",
        n_bands=n_bands,
    )
    # image.normalize()
    mask = labeler.create_segmentation_label(image)
    print(np.sum(mask))
    if np.sum(mask) > 0:
        lsi = SegmentationLabeledSatelliteImage(image, mask)
        image_mask = lsi.plot(bands_indices=[0, 1, 2])

        filename = im.split("/")[-1]
        filename = filename.split(".")[0]
        image_mask.savefig(
            f"../data/affichage_data_raw/{type_labeler}/{task}/{source}/{dep}/{year}/{filename}.png"
        )
        plt.close()
