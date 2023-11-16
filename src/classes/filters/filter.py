import geopandas as gpd
import numpy as np
from astrovision import SatelliteImage
from rasterio.features import rasterize, shapes
from shapely.geometry import Polygon
from skimage.measure import label

# from skimage.draw import polygon


class Filter:
    """
    Filter class.
    """

    def __init__(self, image: SatelliteImage):
        self.image = image

    def is_too_black(self, black_value_threshold=100, black_area_threshold=0.5) -> bool:
        """
        Determine if a satellite image is too black
        based on pixel values and black area proportion.

        This function converts a satellite image to grayscale and
        filters it based on the number of black pixels and their proportion.
        A pixel is considered black if its value is less than the specified
        threshold (black_value_threshold).

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
        gray_image = 0.2989 * self.array[0] + 0.5870 * self.array[1] + 0.1140 * self.array[2]
        nb_black_pixels = np.sum(gray_image < black_value_threshold)

        if (nb_black_pixels / (gray_image.shape[0] ** 2)) >= black_area_threshold:
            return True
        else:
            return False

    def mask_cloud(self, threshold: float = 0.98, min_size: int = 50000) -> np.ndarray:
        """
        Detects clouds in a SatelliteImage using a threshold-based approach
        (grayscale threshold and pixel cluster size threshold) and
        returns a binary mask of the detected clouds.
        This function works on RGB images in the format (C x H x W)
        encoded in float.

        Args:
            image (SatelliteImage):
                The input satellite image to process.
            threshold (int):
                The threshold value to use for detecting clouds on the image
                transformed into grayscale. A pixel is considered part of a
                cloud if its value is greater than this threshold.
                Default to 0.98.
            min_size (int):
                The minimum size (in pixels) of a cloud region to be
                considered valid.
                Default to 50000.

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
        copy_image = self.copy()

        if not copy_image.normalized:
            copy_image.normalize()

        image = copy_image.array
        image = image[[0, 1, 2], :, :]
        image = (image * np.max(image)).astype(np.float64)
        image = image.transpose(1, 2, 0)

        # Convert the RGB image to grayscale
        grayscale = np.mean(image, axis=2)

        # Find clusters of white pixels that correspond to 5% or more of the image
        labeled, num_features = label(grayscale > threshold)

        region_sizes = np.bincount(labeled.flat)

        # Sort region labels by decreasing size
        sorted_labels = np.argsort(-region_sizes)

        # Minimum size of the cluster
        mask = np.zeros_like(labeled)

        if num_features >= 1:
            # Display the progress bar
            # for i in tqdm(range(1, num_features + 1)):
            for i in range(1, num_features + 1):
                if region_sizes[sorted_labels[i]] >= min_size:
                    mask[labeled == sorted_labels[i]] = 1
                else:
                    break

        # Return the cloud mask
        return mask

    def mask_full_cloud(
        self,
        threshold_center: float = 0.98,
        threshold_full: float = 0.7,
        min_size: int = 50000,
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
            Defaults to 0.98 (white pixels).
        threshold_full (int, optional):
            An integer representing the threshold for coverage of the full clouds
            in the image. Pixels with a cloud coverage value higher than this
            threshold are classified as covered by clouds.
            Defaults to 0.7 (light grey pixels).
        min_size (int, optional):
            An integer representing the minimum size (in pixels) of a cloud region
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
            >>> mask_full = mask_full_cloud(image_1)
            >>> fig, ax = plt.subplots(figsize=(10, 10))
            >>> ax.imshow(np.transpose(image_1.array, (1, 2, 0))[:,:,:3])
            >>> ax.imshow(mask_full, alpha=0.3)
        """
        # Mask out clouds from the image using different thresholds
        cloud_center = self.mask_cloud(threshold_center, min_size)
        cloud_full = self.mask_cloud(threshold_full, min_size)

        nchannel, height, width = self.array.shape

        # Create a list of polygons from the masked center clouds in order
        # to obtain a GeoDataFrame from it
        polygon_list_center = []
        for shape in list(shapes(cloud_center)):
            polygon = Polygon(shape[0]["coordinates"][0])
            if polygon.area > 0.85 * height * width:
                continue
            polygon_list_center.append(polygon)

        g_center = gpd.GeoDataFrame(geometry=polygon_list_center)

        # Same but from the masked full clouds
        polygon_list_full = []
        for shape in list(shapes(cloud_full)):
            polygon = Polygon(shape[0]["coordinates"][0])
            if polygon.area > 0.85 * height * width:
                continue
            polygon_list_full.append(polygon)

        g_full = gpd.GeoDataFrame(geometry=polygon_list_full)

        # Spatial join on the GeoDataFrames for the masked full clouds
        # and the masked center clouds
        result = gpd.sjoin(g_full, g_center, how="inner", predicate="intersects")

        # Remove any duplicate geometries
        result = result.drop_duplicates(subset="geometry")

        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(np.transpose(image.array, (1, 2, 0))[:,:,:3])
        # result.plot(color = "orange", ax=ax)

        # Rasterize the geometries into a numpy array
        if result.empty:
            rasterized = np.zeros(self.array.shape[1:])
        else:
            rasterized = rasterize(
                result.geometry,
                out_shape=self.array.shape[1:],
                fill=0,
                out=None,
                all_touched=True,
                default_value=1,
                dtype=None,
            )

        return rasterized
