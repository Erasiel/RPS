from PyQt5 import QtGui, QtWidgets
import cv2
import sys
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from DataCollector import DataCollector


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def grab_mid(self, cv_img, x, y):
        img_x, img_y, _ = cv_img.shape
        x_cutoff = (img_x - x) // 2
        y_cutoff = (img_y - y) // 2
        return cv_img[x_cutoff:x_cutoff + x, y_cutoff:y_cutoff+y, :]

    def run(self):
        # Capture from webcam
        # IMPORTANT: that integer parameter is extremely magical
        # It's the number of the USB port to which the webcam is connected
        # or 0 if the cam is integrated.
        cap = cv2.VideoCapture(1)

        while True:
            ret, cv_img = cap.read()
            if ret:
                mid_image = self.grab_mid(cv_img, 256, 256)
                self.change_pixmap_signal.emit(mid_image)


class DataCollectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DataCollectionWidget, self).__init__(parent)

        # Create data collector
        self.data_collector = DataCollector(".")

        # Setup image container
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Create image capture thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

        # Setup train-validation checkboxes
        self.train_cb = QtWidgets.QCheckBox("Train", self)
        self.validation_cb = QtWidgets.QCheckBox("Validation", self)
        self.train_cb.stateChanged.connect(self.update_check)
        self.validation_cb.stateChanged.connect(self.update_check)
        self.train_cb.setChecked(True)
        self.target_folder = "train"

        checkbox_layout = QtWidgets.QHBoxLayout()
        checkbox_layout.addWidget(self.train_cb)
        checkbox_layout.addWidget(self.validation_cb)

        # Setup buttons
        rock_button = QtWidgets.QPushButton('ROCK')
        paper_button = QtWidgets.QPushButton('PAPER')
        scissors_button = QtWidgets.QPushButton('SCISSORS')
        none_button = QtWidgets.QPushButton('NONE')
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(rock_button)
        button_layout.addWidget(paper_button)
        button_layout.addWidget(scissors_button)
        button_layout.addWidget(none_button)

        # Connect button event handlers
        rock_button.clicked.connect(lambda: self.save_image("ROCK"))
        paper_button.clicked.connect(lambda: self.save_image("PAPER"))
        scissors_button.clicked.connect(lambda: self.save_image("SCISSORS"))
        none_button.clicked.connect(lambda: self.save_image("NONE"))

        # Setup layout
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(checkbox_layout)
        self.layout.addLayout(button_layout)
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
        self.setWindowTitle("Data collection")

        # Image content related variables
        self.display_width = 256
        self.display_height = 256
        self.cv_img = None

        # Start image capture
        self.thread.start()

    def update_check(self, state):
        if state == Qt.Checked:
            if (self.sender() == self.train_cb):
                self.target_folder = "train"
                self.validation_cb.setChecked(False)
            else:
                self.target_folder = "validation"
                self.train_cb.setChecked(False)

    def save_image(self, label):
        self.data_collector.save_image(self.cv_img, label, self.target_folder)

    def update_image(self, cv_img):
        self.cv_img = cv_img
        qt_img = self.convert_cv_qt_color(cv_img)
        # qt_img = self.convert_cv_qt_gray(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt_color(self, cv_img):
        """Convert from a BGR opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        convert_to_Qt_format = QtGui.QImage(rgb_image.data,
                                            w,
                                            h,
                                            bytes_per_line,
                                            QtGui.QImage.Format_RGB888)

        p = convert_to_Qt_format.scaled(self.display_width,
                                        self.display_height,
                                        Qt.KeepAspectRatio)

        return QtGui.QPixmap.fromImage(p)

    def convert_cv_qt_gray(self, cv_img):
        """Convert from a grayscale opencv image to QPixmap"""
        self.cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = rgb_image.shape
        bytes_per_line = w

        convert_to_Qt_format = QtGui.QImage(rgb_image.data,
                                            w,
                                            h,
                                            bytes_per_line,
                                            QtGui.QImage.Format_Grayscale8)

        p = convert_to_Qt_format.scaled(self.display_width,
                                        self.display_height,
                                        Qt.KeepAspectRatio)

        return QtGui.QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    display_image_widget = DataCollectionWidget()
    display_image_widget.show()
    sys.exit(app.exec_())
