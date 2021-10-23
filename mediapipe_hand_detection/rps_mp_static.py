# ------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
import sys
from pathlib import Path

# ------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------

def get_landmark_coordinates(mp_hands, hand_landmarks, width, height):
    """
    Return the coordinates of the hand landmarks in a dictionary where the key is the name
    of the landmark and the value is another dictionary with "x" and "y" keys with the
    normalized coordinates values.
    """
    if hand_landmarks:
        return {
            landmark.name: {
                "x": hand_landmarks.landmark[landmark].x * width,
                "y": hand_landmarks.landmark[landmark].y * height
            } for landmark in mp_hands.HandLandmark
        }
    else:
        return None

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------

# Mediapipe utility variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Read the first command line argument (should contain the path of an image of a hand)
if len(sys.argv) > 1:
    file = Path(sys.argv[1])

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    # Read and process image and get detection output
    image = cv2.imread(str(file))
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Get image properties
    height, width, _ = image.shape

    # Draw the landmarks on the image and print the landmark coordinates
    annotated_image = image.copy()

    for hand_landmarks in results.multi_hand_landmarks:
        landmark_coords = get_landmark_coordinates(mp_hands, hand_landmarks, width, height)

        print("Landmark coordinates:")
        for (name, coords) in landmark_coords.items():
            print(f"{name}: ({coords['x']}, {coords['y']})")

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Save the annotated image
    save_path = file.parent / (file.stem + "_ANNOTATED" + ".png")
    cv2.imwrite(str(save_path), annotated_image)
