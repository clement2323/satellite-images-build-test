import os
import re
from astrovision.data import SatelliteImage
from astrovision.plot import plot_images
from tqdm import tqdm

from functions import download_data

dep = "MAYOTTE"
source = "PLEIADES"
year = "2022"
task = "segmentation"
type_labeler = "BDTOPO"
tiles_size = "250"
n_bands = "3"
# Initialize S3 file system
fs = download_data.get_file_system()

image_filepaths = fs.ls(
    f"projet-slums-detection/data-preprocessed/patchs/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/train"
)


def group_patchs_of_images(image_filepaths):
    grouped = {}
    for filepath in image_filepaths:
        delimiters = ["-", "_"]
        pattern = "|".join(delimiters)
        split_filepath = filepath.split("/")
        filename = split_filepath[-1]
        begin_filepath = ("/").join(split_filepath[:-1])
        split_filename = re.split(pattern, filename)
        suffix = split_filename[-1]

        if "_" in filename:
            prefix = ("_").join(split_filename[:-1])
        elif "-" in filename:
            prefix = ("-").join(split_filename[:-1])
        prefix_filepath = f"{begin_filepath}/{prefix}.jp2"

        if suffix.split(".")[-1] == "jp2":
            if prefix_filepath not in grouped:
                grouped[prefix_filepath] = []  # Initialiser la liste si nécessaire
            grouped[prefix_filepath].append(filepath)
    return grouped


grouped_filepaths = group_patchs_of_images(image_filepaths)

# Affichage des résultats

os.makedirs(
    f"../data/affichage_patchs_preprocessed/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/train/",  # noqa
    exist_ok=True,
)

for image_filepaths in list(grouped_filepaths.keys()):
    list_to_plot = grouped_filepaths[image_filepaths]

    liste_images = []
    for im in tqdm(list_to_plot):
        # 1- Ouvrir avec SatelliteImage
        image = SatelliteImage.from_raster(
            file_path=f"/vsis3/{im}",
            n_bands=n_bands,
        )
        image.normalize()
        liste_images.append(image)

    images = plot_images(liste_images, bands_indices=[0, 1, 2])
    filename = image_filepaths.split("/")[-1]
    filename = filename.split(".")[0]
    images.savefig(
        f"../data/affichage_patchs_preprocessed/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/train/{filename}.png"
    )
