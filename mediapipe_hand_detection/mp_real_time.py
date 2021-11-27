# ------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------

import cv2
import joblib
import numpy as np
import rps_utils as rps

# ------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------

MODEL_PATH = "./model_1637688558.pkl"

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Load classifier model
    clf = joblib.load(MODEL_PATH)

    # Get webcam as source
    capture = cv2.VideoCapture(0)

    # Real time hand detection
    with rps.MP_HANDS.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while capture.isOpened():
            success, image = capture.read()

            # Test rps.detect_gesture function
            predicted_label_function_test = rps.detect_gesture(image)
            print(f"FUNCTION TEST: {predicted_label_function_test}")

            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Read and flip image
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # Process image and get detection output
            image.flags.writeable = False
            results = hands.process(image)

            # Get image properties
            height, width, _ = image.shape

            # Make image writeable
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks is not None:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Predict gesture
                landmark_coords = rps.get_landmark_coordinates(hand_landmarks, width, height)
                landmark_dists = rps.landmark_distances(landmark_coords)
                dist_features = np.array(landmark_dists).reshape(1, -1)
                predicted_label = clf.predict(dist_features)[0]
                print(f"PREDICTION: {rps.LABELS[predicted_label]}\n")

                # Draw the landmarks on the image
                rps.MP_DRAW.draw_landmarks(
                    image,
                    hand_landmarks,
                    rps.MP_HANDS.HAND_CONNECTIONS,
                    rps.MP_DRAW_STYLES.get_default_hand_landmarks_style(),
                    rps.MP_DRAW_STYLES.get_default_hand_connections_style())

            # Show image
            cv2.imshow("Rock-paper-scissors", image)

            # Close webcam capture when 'esc' is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Release capture
    capture.release()
