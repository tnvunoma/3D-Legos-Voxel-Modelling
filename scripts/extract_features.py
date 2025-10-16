import cv2
import numpy as np

def extract_features_and_match(image_paths):
    sift = cv2.SIFT.create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    keypoints_all = []
    descriptors_all = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            keypoints_all.append([])
            descriptors_all.append(None)
            continue
        keypoints, descriptors = sift.detectAndCompute(image, None) # type: ignore
        keypoints_all.append(keypoints)
        descriptors_all.append(descriptors)

    matches_all = []
    for i in range(len(image_paths) - 1):
        if descriptors_all[i] is None or descriptors_all[i+1] is None:
            matches_all.append([])
            continue
        matches = bf.match(descriptors_all[i], descriptors_all[i+1])
        matches = sorted(matches, key=lambda x: x.distance)
        matches_all.append(matches)

    return keypoints_all, descriptors_all, matches_all


# Example usage
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
keypoints_all, descriptors_all, matches_all = extract_features_and_match(image_paths)
