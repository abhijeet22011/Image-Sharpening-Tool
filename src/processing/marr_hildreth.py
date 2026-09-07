import cv2
import numpy as np


def detect_marr_hildreth_edges(
    image,
    sigma=1.4,
    threshold=15.0
):
    """
    Detect edges using the Marr-Hildreth
    Laplacian of Gaussian (LoG) method.

    Parameters:
        image: Input BGR image
        sigma: Gaussian standard deviation
        threshold: Minimum LoG response difference
                   required for a zero crossing

    Returns:
        Binary edge image
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Gaussian smoothing
    kernel_size = int(6 * sigma + 1)

    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(
        gray,
        (kernel_size, kernel_size),
        sigma
    )

    # Laplacian
    laplacian = cv2.Laplacian(
        blurred,
        cv2.CV_64F
    )

    # Compare neighboring pixels
    center = laplacian[1:-1, 1:-1]

    neighbors = [
        laplacian[:-2, :-2],
        laplacian[:-2, 1:-1],
        laplacian[:-2, 2:],
        laplacian[1:-1, :-2],
        laplacian[1:-1, 2:],
        laplacian[2:, :-2],
        laplacian[2:, 1:-1],
        laplacian[2:, 2:]
    ]

    # Find maximum and minimum neighbor values
    max_neighbor = np.maximum.reduce(neighbors)
    min_neighbor = np.minimum.reduce(neighbors)

    # Detect zero crossings
    zero_crossing = (
        ((center > 0) & (min_neighbor < 0)) |
        ((center < 0) & (max_neighbor > 0))
    )

    # Calculate local LoG response difference
    response = np.maximum(
        np.abs(center - max_neighbor),
        np.abs(center - min_neighbor)
    )

    # Apply threshold
    edges = (
        zero_crossing &
        (response >= threshold)
    )

    # Convert to 8-bit image
    edge_image = np.zeros_like(
        laplacian,
        dtype=np.uint8
    )

    edge_image[1:-1, 1:-1][edges] = 255

    return edge_image