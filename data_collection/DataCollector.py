import os
import cv2
import time
import calendar


def create_directory(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


class DataCollector:
    def __init__(self, base_path="."):
        # Base directory for all data
        data_dir = os.path.join(base_path, "data")

        # Train and validation data go in separate directories
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "validation")

        # Train data directories
        train_rock_dir = os.path.join(train_dir, "rock")
        train_paper_dir = os.path.join(train_dir, "paper")
        train_scissors_dir = os.path.join(train_dir, "scissors")
        train_none_dir = os.path.join(train_dir, "none")

        # Validation data directories
        val_rock_dir = os.path.join(val_dir, "rock")
        val_paper_dir = os.path.join(val_dir, "paper")
        val_scissors_dir = os.path.join(val_dir, "scissors")
        val_none_dir = os.path.join(val_dir, "none")

        create_directory(data_dir)
        create_directory(train_dir)
        create_directory(val_dir)
        create_directory(train_rock_dir)
        create_directory(train_paper_dir)
        create_directory(train_scissors_dir)
        create_directory(train_none_dir)
        create_directory(val_rock_dir)
        create_directory(val_paper_dir)
        create_directory(val_scissors_dir)
        create_directory(val_none_dir)

        self.validation_dir = val_dir
        self.train_dir = train_dir

    def save_image(self, image, label, target_folder):
        label = label.lower()
        gmt = time.gmtime()
        timestamp = calendar.timegm(gmt)

        filename = f"{label}{timestamp}.jpg"

        if target_folder == "train":
            filepath = os.path.join(self.train_dir, label, filename)
        else:
            filepath = os.path.join(self.validation_dir, label, filename)

        cv2.imwrite(filepath, image)
        print(f"Succesfully saved {filepath}")
