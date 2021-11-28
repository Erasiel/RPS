import cv2
from keras import models
import numpy as np

model = models.load_model("rps_transfer_model.h5")

PREDICTION_LABELS = {
    0: "paper",
    1: "rock",
    2: "scissors"
}


def predict(image) -> str:
    global model

    # change to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # resize image
    image = cv2.resize(image, (128, 128))

    # scale image values
    image = image / 255.0

    pred = model.predict(np.array([image]))

    return PREDICTION_LABELS[np.argmax(pred[0])]


def grab_mid(cv_img, x, y):
        img_x, img_y, _ = cv_img.shape
        x_cutoff = (img_x - x) // 2
        y_cutoff = (img_y - y) // 2
        return cv_img[x_cutoff:x_cutoff + x, y_cutoff:y_cutoff+y, :]


if __name__ == "__main__":
    vid = cv2.VideoCapture(1)
    counter = 0
  
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        frame = grab_mid(frame, 256, 256)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        counter += 1

        if counter % 1 == 0:
            print(predict(frame))
            counter = 0


        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    # scissors = cv2.imread("../data_collection/data/train/scissors/scissors1636310467.jpg")
    # predict(scissors)

    # rock = cv2.imread("../data_collection/data/train/rock/rock1636316027.jpg")
    # predict(rock)
    
    # paper = cv2.imread("../data_collection/data/train/paper/paper1636310527.jpg")
    # predict(paper)
