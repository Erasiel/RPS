# ------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------

# Mediapipe utility variables
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Get webcam as source
capture = cv2.VideoCapture(0)

# Real time hand detection

print("Press ESC to exit.")

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while capture.isOpened():
        success, image = capture.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Read and flip image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process image and get detection output
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the landmarks on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Show image
        cv2.imshow("Rock-paper-scissors", image)

        # Close webcam capture when 'esc' is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release capture
capture.release()
