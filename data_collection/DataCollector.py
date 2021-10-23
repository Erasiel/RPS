import os
import cv2
import time
import calendar


def create_directory(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


class DataCollector:
    def __init__(self, base_path="."):
        data_dir = os.path.join(base_path, "data")
        train_dir = os.path.join(data_dir, "train")
        rock_dir = os.path.join(train_dir, "rock")
        paper_dir = os.path.join(train_dir, "paper")
        scissors_dir = os.path.join(train_dir, "scissors")
        none_dir = os.path.join(train_dir, "none")

        create_directory(data_dir)
        create_directory(train_dir)
        create_directory(rock_dir)
        create_directory(paper_dir)
        create_directory(scissors_dir)
        create_directory(none_dir)

        self.train_dir = train_dir

    def save_image(self, image, label):
        label = label.lower()
        gmt = time.gmtime()
        timestamp = calendar.timegm(gmt)
        
        filename = f"{label}{timestamp}.jpg"
        filepath = os.path.join(self.train_dir, label, filename)
        
        cv2.imwrite(filepath, image)
        print(f"Succesfully saved {filepath}")
