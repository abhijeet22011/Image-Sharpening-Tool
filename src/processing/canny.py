import cv2


def detect_canny_edges(image, low_threshold=100, high_threshold=200):
    """
    Detect edges using the Canny edge detection algorithm.

    Parameters:
        image: Input image
        low_threshold: Lower threshold for Canny
        high_threshold: Upper threshold for Canny

    Returns:
        Edge image
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(
        blurred,
        low_threshold,
        high_threshold
    )

    return edges