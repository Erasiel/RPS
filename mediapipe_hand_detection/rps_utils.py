# ------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------

import cv2
import joblib
import math
import mediapipe as mp
import numpy as np
import os
import pathlib

# ------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------

# Mediapipe utilities
MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils
MP_DRAW_STYLES = mp.solutions.drawing_styles

# Path of the trained model
MODEL_PATH = "./model_1637688558.pkl"

# Labels for rps gestures
LABELS = {
    0: "Rock",
    1: "Paper",
    2: "Scissors",
    3: "None"
}

# ------------------------------------------------------------------------------------------
# Implementation
# ------------------------------------------------------------------------------------------

def detect_gesture(image):
    """
    Process the the image and determine the hand gesture with the help of the trained model.
    Return the name of the gesture (if no hand is detected, return "None" as a string).
    """
    # Load trained model
    clf = joblib.load(MODEL_PATH)

    # Run mediapipe hand detection
    with MP_HANDS.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        # Change color model
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process image and get detection output
        image.flags.writeable = False
        results = hands.process(image)

        # Get image properties
        height, width, _ = image.shape

        if results.multi_hand_landmarks is not None:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Predict gesture
            landmark_coords = get_landmark_coordinates(hand_landmarks, width, height)
            landmark_dists = landmark_distances(landmark_coords)
            dist_features = np.array(landmark_dists).reshape(1, -1)
            predicted_label = clf.predict(dist_features)[0]
            return LABELS[predicted_label]

        return "None"

def process_images(dir_path, label):
    """
    Process all of the images in the given directory and save the results into a csv file
    where rows are the samples and columns are the features and each sample contains the
    label in the first column.
    """
    image_paths = [dir_path / f for f in os.listdir(dir_path) if f.endswith(".jpg")]

    results = []
    for image_path in image_paths:
        image_result = process_image(image_path)
        if image_result is not None:
            results.append([label] + image_result)

    csv_path = str(dir_path / (dir_path.stem + ".csv"))
    np.savetxt(csv_path, results, delimiter=';')

def process_image(file_path):
    """
    Run hand detection on the given image and create an annotated image and return a vector
    with the distance values. Return None if no hand is detected.
    """
    with MP_HANDS.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        # Run detection and get the landmark values
        image = cv2.imread(str(file_path))
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Get image properties
        height, width, _ = image.shape

        # Calculate distance values if the detection was successful
        if results.multi_hand_landmarks is not None:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmark_coords = get_landmark_coordinates(hand_landmarks, width, height)
            return landmark_distances(landmark_coords)

        return None

def landmark_distances(landmark_coordinates):
    """
    Calculate the distance values between all landmarks.
    """
    coord_list = list(landmark_coordinates.values())

    distances = []
    for i in range(len(landmark_coordinates)):
        for j in range(i + 1, len(landmark_coordinates)):
            distances.append(euclidean_dist(coord_list[i]["x"],
                                            coord_list[j]["x"],
                                            coord_list[i]["y"],
                                            coord_list[j]["y"]))
    return distances

def get_landmark_coordinates(hand_landmarks, width, height, multiplier = 256):
    """
    Return the coordinates of the hand landmarks in a dictionary where the key is the name
    of the landmark and the value is another dictionary with the keys "x" and "y" and the
    normalized coordinates as values.
    """
    if hand_landmarks:
        norm_landmarks = {
            landmark.name: {
                "x": hand_landmarks.landmark[landmark].x * width,
                "y": hand_landmarks.landmark[landmark].y * height
            } for landmark in MP_HANDS.HandLandmark
        }

        min_x = min([lm["x"] for lm in norm_landmarks.values()])
        max_x = max([lm["x"] for lm in norm_landmarks.values()])
        min_y = min([lm["y"] for lm in norm_landmarks.values()])
        max_y = max([lm["y"] for lm in norm_landmarks.values()])

        return {
            name: {
                "x": minmax_norm(coords["x"], min_x, max_x, multiplier),
                "y": minmax_norm(coords["y"], min_y, max_y, multiplier)
            } for (name, coords) in norm_landmarks.items()
        }
    else:
        return None

# ------------------------------------------------------------------------------------------
# Internals
# ------------------------------------------------------------------------------------------

def minmax_norm(value, min, max, multiplier = 256):
    """
    Perform min-max normalization.
    """
    return (value - min) / (max - min) * multiplier

def euclidean_dist(x1, x2, y1, y2):
    """
    Calculate euclidean distance.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
