import math

import matplotlib.pyplot as plt
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
from tqdm import tqdm

from classes.filters.filter import Filter


def plot_list_path_square(list_filepaths: list, filter_, source, dep, year):
    size = int(math.sqrt(len(list_filepaths)))
    bands_indices = [0, 1, 2]

    list_labeled_image = []

    for im in tqdm(list_filepaths):
        # 1- Ouvrir avec SatelliteImage
        image = SatelliteImage.from_raster(
            file_path=f"../data/data-raw/{dep}/{year}/{im}",
            n_bands=len(bands_indices),
        )
        image.normalize()

        mask_cloud = filter_.create_mask_cloud(image, 0.7, 0.4, 0.0125)
        lsi = SegmentationLabeledSatelliteImage(image, mask_cloud)
        list_labeled_image.append(lsi)

        # plt.imshow(mask_cloud, cmap = 'gray')

    list_images1 = [iml.satellite_image for iml in list_labeled_image]
    list_labels1 = [iml.label for iml in list_labeled_image]

    list_bounding_box = [[im.bounds[3], im.bounds[0]] for im in list_images1]

    # Utiliser zip pour combiner les trois listes
    combined = zip(list_bounding_box, list_images1, list_labels1)

    # Trier les éléments combinés en fonction de la troisième liste
    sorted_combined = sorted(combined, key=lambda x: (-x[0][0], x[0][1]))

    # Diviser les listes triées en fonction de l'ordre des éléments
    __, list_images, list_labels = zip(*sorted_combined)

    size = int(math.sqrt(len(list_images)))

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=size, ncols=2 * size, figsize=(20, 10))

    # Iterate over the grid of masks and plot them
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(
                list_images[i * size + j].array.transpose(1, 2, 0)[:, :, bands_indices]
            )

    for i in range(size):
        for j in range(size):
            axs[i, j + size].imshow(list_labels[i * size + j], cmap="gray")

    # Remove any unused axes
    for i in range(size):
        for j in range(2 * size):
            axs[i, j].set_axis_off()

    # Show the plot
    return plt.gcf()


def plot_square_nb_images_folder(source, dep, year, filter_: Filter, debut, fin, n_bands, fs):
    bands_indices = [i for i in range(int(n_bands))]

    list_labeled_image = []

    list_images = fs.ls(f"projet-slums-detection/data-raw/{source}/{dep}/{year}/")[debut:fin]
    size = int(math.sqrt(len(list_images)))

    for im_path in tqdm(list_images):
        # 1- Ouvrir avec SatelliteImage
        image = SatelliteImage.from_raster(
            file_path=f"s3://{im_path}",
            n_bands=int(n_bands),
        )
        image.normalize()

        mask_cloud = filter_.create_mask_cloud(image, 0.7, 0.4, 0.0125)
        lsi = SegmentationLabeledSatelliteImage(image, mask_cloud)
        list_labeled_image.append(lsi)

        # plt.imshow(mask_cloud, cmap = 'gray')

    list_images1 = [iml.satellite_image for iml in list_labeled_image]
    list_labels1 = [iml.label for iml in list_labeled_image]

    list_bounding_box = [[im.bounds[3], im.bounds[0]] for im in list_images1]

    # Utiliser zip pour combiner les trois listes
    combined = zip(list_bounding_box, list_images1, list_labels1)

    # Trier les éléments combinés en fonction de la troisième liste
    sorted_combined = sorted(combined, key=lambda x: (-x[0][0], x[0][1]))

    # Diviser les listes triées en fonction de l'ordre des éléments
    __, list_images, list_labels = zip(*sorted_combined)

    size = int(math.sqrt(len(list_images)))

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=size, ncols=2 * size, figsize=(20, 10))

    # Iterate over the grid of masks and plot them
    for i in range(size):
        for j in range(size):
            axs[i, j].imshow(
                list_images[i * size + j].array.transpose(1, 2, 0)[:, :, bands_indices]
            )

    for i in range(size):
        for j in range(size):
            axs[i, j + size].imshow(list_labels[i * size + j], cmap="gray")

    # Remove any unused axes
    for i in range(size):
        for j in range(2 * size):
            axs[i, j].set_axis_off()

    # Show the plot
    return plt.gcf()
