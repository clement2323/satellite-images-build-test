import geopandas as gpd
import numpy as np
from astrovision.data import SatelliteImage
from astrovision.data.utils import generate_tiles_borders
from rasterio.features import rasterize, shapes
from scipy.ndimage import label
from shapely.geometry import Polygon


class Filter:
    """
    Filter class.
    """

    def __init__(self):
        return

    def is_too_black(
        self, image: SatelliteImage, black_value_threshold=100, black_area_threshold=0.5
    ) -> bool:
        """
        Determine if a satellite image is too black
        based on pixel values and black area proportion.

        This function converts a satellite image to grayscale and
        filters it based on the number of black pixels and their proportion.
        A pixel is considered black if its value is less than the specified
        threshold (black_value_threshold).
        formula used : 0.2989red + 0.587green + 0.114blue
        The image is considered too black if the proportion of black pixels
        is greater than or equal to the specified threshold (black_area_threshold).

        Args:
            image (SatelliteImage): The input satellite image.
            black_value_threshold (int, optional): The threshold value
                for considering a pixel as black. Default is 100.
            black_area_threshold (float, optional): The threshold for
                the proportion of black pixels. Default is 0.5.

        Returns:
            bool: True if the proportion of black pixels is greater than or equal
                to the threshold, False otherwise.
        """
        gray_image = 0.2989 * image.array[0] + 0.5870 * image.array[1] + 0.1140 * image.array[2]
        nb_black_pixels = np.sum(gray_image < black_value_threshold)

        if (nb_black_pixels / np.prod(gray_image.shape)) >= black_area_threshold:
            return True
        else:
            return False

    def mask_cloud(
        self, image: SatelliteImage, threshold: float = 0.98, min_relative_size: float = 0.0125
    ) -> np.ndarray:
        """
        Detects clouds in a SatelliteImage using a threshold-based approach
        (grayscale threshold and pixel cluster size threshold) and
        returns a binary mask of the detected clouds.
        This function works on RGB images in the format (C x H x W)
        encoded in float.

        Args:
            image (SatelliteImage):
                The input satellite image to process.
            threshold (float):
                The threshold value to use for detecting clouds on the image
                transformed into grayscale. A pixel is considered part of a
                cloud if its value is greater than this threshold.
                Default to 0.98.
            min_relative_size (float):
                The minimum relative size (in pixels) of a cloud region to be
                considered valid.
                Default to 1.25%.

        Returns:
            mask (np.ndarray):
                A binary mask of the detected clouds in the input image.

        Example:
            >>> filename_1 = '../data/PLEIADES/2020/MAYOTTE/
            ORT_2020052526656219_0508_8599_U38S_8Bits.jp2'
            >>> date_1 = date.fromisoformat('2020-01-01')
            >>> image_1 = SatelliteImage.from_raster(
                                        filename_1,
                                        date = date_1,
                                        n_bands = 3,
                                        dep = "976"
                                    )
            >>> mask = mask_cloud(image_1)
            >>> fig, ax = plt.subplots(figsize=(10, 10))
            >>> ax.imshow(np.transpose(image_1.array, (1, 2, 0))[:,:,:3])
            >>> ax.imshow(mask, alpha=0.3)
        """
        # Convert the RGB image to grayscale
        weights = np.array([0.2989, 0.5870, 0.1140])[:, np.newaxis, np.newaxis]
        grayscale = np.sum(weights * image.array, axis=0)
        grayscale = grayscale.astype(image.array.dtype)

        # Compute absolute threshold
        if grayscale.dtype == np.uint8:
            absolute_threshold = threshold * (2**8 - 1)
        elif grayscale.dtype == np.uint16:
            absolute_threshold = threshold * (2**16 - 1)
        else:
            raise ValueError(
                f"Unsupported dtype: {grayscale.dtype}. Expected np.uint8 or np.uint16."
            )

        # Find clusters of white pixels
        labeled, num_features = label(grayscale > absolute_threshold)
        region_sizes = np.bincount(labeled.flat)

        # Sort region labels by decreasing size
        sorted_labels = np.argsort(-region_sizes)

        mask = np.zeros_like(labeled)
        if num_features >= 1:
            for i in range(1, num_features + 1):
                # Minimum size of the cluster
                if region_sizes[sorted_labels[i]] >= min_relative_size * np.prod(grayscale.shape):
                    mask[labeled == sorted_labels[i]] = 1
                else:
                    break

        # Return the cloud mask
        return mask

    def create_mask_cloud(
        self,
        image: SatelliteImage,
        threshold_center: float = 0.7,
        threshold_full: float = 0.4,
        min_relative_size: float = 0.0125,
    ) -> np.ndarray:
        """
        Masks out clouds in a SatelliteImage using two thresholds for cloud
        coverage, and returns the resulting cloud mask as a numpy array.

        Parameters:
        -----------
        image (SatelliteImage):
            An instance of the SatelliteImage class representing the input image
            to be processed.
        threshold_center (int, optional):
            An integer representing the threshold for coverage of the center of
            clouds in the image. Pixels with a cloud coverage value higher than
            this threshold are classified as cloud-covered.
            Defaults to 0.7 (white pixels).
        threshold_full (int, optional):
            An integer representing the threshold for coverage of the full clouds
            in the image. Pixels with a cloud coverage value higher than this
            threshold are classified as covered by clouds.
            Defaults to 0.4 (light grey pixels).
        min_relative_size (float, optional):
            An integer representing the minimum relative size (in pixels) of a cloud region
            that will be retained in the output mask.
            Defaults to 50,000 (2,000*2,000 = 4,000,000 pixels and we want to
            detect clouds that occupy > 1.25% of the image).

        Returns:
        --------
        rasterized (np.ndarray):
            A numpy array representing the rasterized version of the cloud mask.
            Pixels with a value of 1 are classified as cloud-free, while pixels
            with a value of 0 are classified as cloud-covered.

        Example:
            >>> filename_1 = '../data/PLEIADES/2020/MAYOTTE/
            ORT_2020052526656219_0508_8599_U38S_8Bits.jp2'
            >>> date_1 = date.fromisoformat('2020-01-01')
            >>> image_1 = SatelliteImage.from_raster(
                                        filename_1,
                                        date = date_1,
                                        n_bands = 3,
                                        dep = "976"
                                    )
            >>> mask_full = create_mask_cloud(image_1)
            >>> fig, ax = plt.subplots(figsize=(10, 10))
            >>> ax.imshow(np.transpose(image_1.array, (1, 2, 0))[:,:,:3])
            >>> ax.imshow(mask_full, alpha=0.3)
        """
        # Mask out clouds from the image using different thresholds
        cloud_center = self.mask_cloud(image, threshold_center, min_relative_size)
        cloud_full = self.mask_cloud(image, threshold_full, min_relative_size)

        # Create a list of polygons from the masked center clouds in order
        # to obtain a GeoDataFrame from it
        g_center = self.mask_to_gdf(cloud_center)

        # Same but from the masked full clouds
        g_full = self.mask_to_gdf(cloud_full)

        # Spatial join on the GeoDataFrames for the masked full clouds
        # and the masked center clouds
        result = gpd.sjoin(g_full, g_center, how="inner", predicate="intersects")

        # Remove any duplicate geometries
        result = result.drop_duplicates(subset="geometry")

        # Rasterize the geometries into a numpy array
        if result.empty:
            rasterized = np.zeros(image.array.shape[1:])
        else:
            rasterized = rasterize(
                result.geometry,
                out_shape=image.array.shape[1:],
                fill=0,
                out=None,
                all_touched=True,
                default_value=1,
                dtype=None,
            )

        return rasterized

    def mask_to_gdf(self, mask: np.array):
        polygon_list = []
        for shape in shapes(mask):
            polygon = Polygon(shape[0]["coordinates"][0])
            if polygon.area > 0.85 * mask.shape[0] * mask.shape[1]:
                continue
            polygon_list.append(polygon)

        gdf = gpd.GeoDataFrame(geometry=polygon_list)
        return gdf

    def is_cloud(
        self,
        si: SatelliteImage,
        tiles_size: int,
        threshold_center: float,
        threshold_full: float,
        min_relative_size: float,
    ):
        mask = self.create_mask_cloud(
            si,
            threshold_center,
            threshold_full,
            min_relative_size,
        )

        indices = generate_tiles_borders(mask.shape[0], mask.shape[1], tiles_size)

        masks = [mask[rows[0] : rows[1], cols[0] : cols[1]] for rows, cols in indices]

        # Return vector of boolean 1 if it is a cloud 0 otherwise
        return [1 if np.sum(mask) > np.prod(mask.shape) * 0.5 else 0 for mask in masks]
