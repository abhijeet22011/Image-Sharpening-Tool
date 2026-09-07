import cv2
import numpy as np


def sharpen_image(image, edges, strength=1.0):
    """
    Sharpen an image using an edge image.

    Formula:
        g(x, y) = f(x, y) + c * e(x, y)

    Parameters:
        image: Original BGR image
        edges: Detected edge image
        strength: Sharpening constant c

    Returns:
        Sharpened image
    """

    # Convert edge image to float
    edge_float = edges.astype(np.float32) / 255.0

    # Convert original image to float
    image_float = image.astype(np.float32)

    # Add scaled edge information
    sharpened = image_float + strength * 255.0 * edge_float[:, :, np.newaxis]

    # Keep pixel values within valid range
    sharpened = np.clip(sharpened, 0, 255)

    # Convert back to uint8
    sharpened = sharpened.astype(np.uint8)

    return sharpened