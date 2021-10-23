import sys
sys.path.insert(1, 'data_collection')

from PyQt5 import QtGui, QtWidgets
import cv2
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from data_collection.DataCollectionWidget import VideoThread


class TitleScreenWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TitleScreenWidget, self).__init__(parent)

        # Image content related variables
        self.display_width = 256
        self.display_height = 256
        self.cv_img = None

        # Setup image containers
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label2 = QtWidgets.QLabel()
        self.image_label2.setAlignment(Qt.AlignCenter)

        # Create image capture thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)

        # Setup buttons
        neural_network_button = QtWidgets.QPushButton('Neural Network')
        media_pipe_button = QtWidgets.QPushButton('MediaPipe')
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(neural_network_button)
        button_layout.addWidget(media_pipe_button)

        # Connect button event handlers
        neural_network_button.clicked.connect(lambda: self.neural_network())
        media_pipe_button.clicked.connect(lambda: self.media_pipe())

        # Member variables for the choices of the players (hard-coded values for the time being)
        player1_choice = 'PAPER'
        player2_choice = 'ROCK'

        # Setup opponent's image as grayscale
        pixmap = QPixmap('sample_images/rock.jpg').scaled(self.display_width, self.display_height)
        q_image = QPixmap.toImage(pixmap)
        grayscale = q_image.convertToFormat(QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(grayscale)
        self.image_label2.setPixmap(pixmap)

        # Setup cameras layout
        camera_layout = QtWidgets.QHBoxLayout()
        camera_layout.addWidget(self.image_label)
        camera_layout.addWidget(self.image_label2)

        # Setup labels
        player1_label = QtWidgets.QLabel(f'Player 1 chose {player1_choice}')
        player2_label = QtWidgets.QLabel(f'Player 2 chose {player2_choice}')
        player1_label.setAlignment(Qt.AlignCenter)
        player2_label.setAlignment(Qt.AlignCenter)
        label_layout = QtWidgets.QHBoxLayout()
        label_layout.addWidget(player1_label)
        label_layout.addWidget(player2_label)

        # Setup layout
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(button_layout)
        self.layout.addLayout(camera_layout)
        self.layout.addLayout(label_layout)
        self.setLayout(self.layout)
        self.setWindowTitle('Rock, Paper, Scissors')
        self.setFixedSize(600, 300)

        # Start image capture
        self.thread.start()

    def neural_network(self):
        """TODO"""
        print('Poof! Doing some neural network magic...')

    def media_pipe(self):
        """TODO"""
        print('Poof! Doing some MediaPipe magic...')

    def update_image(self, cv_img):
        self.cv_img = cv_img
        qt_img = self.convert_cv_qt_gray(cv_img)
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
    title_screen_widget = TitleScreenWidget()
    title_screen_widget.show()
    sys.exit(app.exec_())
