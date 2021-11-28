import sys

from ai_player.easy import get_easy_action
from ai_player.utils import get_winner
sys.path.append('data_collection')

import cv2
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from data_collection.DataCollectionWidget import VideoThread
from ai_player.normal import get_normal_action, update_player1_chances
from ai_player.easy import get_easy_action
from ai_player.hard import get_hard_action


class ChangeDirectory:
    """
    A simple context manager util class that enters and exists a given directory. This will come in handy when
    importing the Neural Network and MediaPipe files.
    """
    def __init__(self, path: str):
        self.stored_path = os.getcwd()                  # Store the path of the current directory
        self.path = path                                # Set the path of the directory to be entered

    def __enter__(self):
        if not os.path.exists(os.path.join(os.getcwd(), self.path)):
            raise RuntimeError(f'The given directory does not exist: {self.path}')

        os.chdir(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.stored_path)


with ChangeDirectory('nn_hand_detection'):
    from nn_hand_detection import homemade_nn


with ChangeDirectory('mediapipe_hand_detection'):
    from mediapipe_hand_detection import rps_utils


class MainWindow(QMainWindow):
    """This class represents the main window of the application."""

    def __init__(self):
        # Call the parent class' constructor
        super().__init__()

        # Set some member variables

        self.window_title = 'Rock, Paper, Scissors'         # The title of the application window
        self.window_width = 600                             # The width of the application window (in pixels)
        self.window_height = 350                            # The height of the application window (in pixels)
        self.difficulty = ''                                # The selected difficulty
        self.detection_method = ''                          # The method used for detecting the hand signs
        self.player1_choice = ''                            # Choice of player 1
        self.player2_choice = ''                            # Choice of player 2

        self.img_width = 256                                # The width of the image representing the computer's choice
        self.img_height = 256                               # The height of the image representing the computer's choice
        self.cv_img = None                                  # The camera image of player 1
        self.cv_img2 = None                                 # The camera image of player 2
        self.thread = None                                  # Image capturing thread

        self.ai_action = get_easy_action                               # AI action function

        # Set up the main window

        self.setWindowTitle(self.window_title)
        self.setFixedSize(self.window_width, self.window_height)

        # Create the three main widgets of the application

        self.left_widget = QWidget()                        # The sidebar widget on the left
        self.right_widget = QWidget()                       # The widget on the right
        self.main_widget = QWidget()                        # The main widget (will contain the left and right widgets)

        # Create the sidebar components

        self.sidebar_label = QLabel('RPS')
        self.sidebar_label.setAlignment(Qt.AlignCenter)
        self.sidebar_label.setStyleSheet('QLabel { font-size: 20px; font-weight: bold; text-align: center; }')

        self.sidebar_btn1 = QPushButton('EASY', self)                                   # Sidebar buttons
        self.sidebar_btn2 = QPushButton('NORMAL', self)
        self.sidebar_btn3 = QPushButton('HARD', self)
        self.sidebar_btn4 = QPushButton('1V1', self)

        self.sidebar_btn1.clicked.connect(lambda: self.switch_difficulty('easy'))       # Add some event handlers
        self.sidebar_btn2.clicked.connect(lambda: self.switch_difficulty('normal'))
        self.sidebar_btn3.clicked.connect(lambda: self.switch_difficulty('hard'))
        self.sidebar_btn4.clicked.connect(lambda: self.switch_difficulty('1v1'))

        # Create "Neural Network" and "MediaPipe" buttons

        self.neural_network_btn = QPushButton('Neural Network')
        self.media_pipe_btn = QPushButton('MediaPipe')
        self.neural_network_btn.clicked.connect(lambda: self.switch_detection_method('neural network'))
        self.media_pipe_btn.clicked.connect(lambda: self.switch_detection_method('mediapipe'))

        # Create the video capturing thread

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # Create the "Capture" buttons for the two players

        self.capture_btn1 = QPushButton('Capture')
        self.capture_btn1.clicked.connect(lambda: self.capture_player1_image())

        # Create the QLabel text element for displaying the result of the game

        self.bottom_text = QLabel()
        self.bottom_text.setText('')
        self.bottom_text.setAlignment(Qt.AlignCenter)
        self.bottom_text.setStyleSheet('QLabel { font-size: 14px; margin-top: 10px; }')

        # Create the QLabel text element for displaying the winner of the game

        self.winner_text = QLabel()
        self.winner_text.setText('')
        self.winner_text.setAlignment(Qt.AlignCenter)
        self.winner_text.setStyleSheet('QLabel { font-size: 14px; margin-top: 10px; }')

        self.switch_difficulty('easy')                                              # Set the default difficulty
        self.switch_detection_method('neural network')                              # Set the default detection method

        # Initialize the UI of the application
        self.init_ui()

    def init_ui(self):
        """Initialize the UI of the application."""

        # === Set up the left widget's layout ===

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.sidebar_label)
        left_layout.addWidget(self.sidebar_btn1)
        left_layout.addWidget(self.sidebar_btn2)
        left_layout.addWidget(self.sidebar_btn3)
        left_layout.addWidget(self.sidebar_btn4)
        left_layout.addStretch(5)
        left_layout.setSpacing(20)
        self.left_widget.setLayout(left_layout)

        # === Set up the right widget's layout ===

        right_layout = QVBoxLayout()

        # Set up top buttons

        top_btn_layout = QHBoxLayout()
        top_btn_layout.addWidget(self.neural_network_btn)
        top_btn_layout.addWidget(self.media_pipe_btn)

        # Set up camera layout

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label2 = QLabel()
        self.image_label2.setAlignment(Qt.AlignCenter)

        self.pixmap = QPixmap('sample_images/rock.jpg').scaled(self.img_width, self.img_height)
        self.image_label2.setPixmap(self.pixmap)

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(self.image_label)
        camera_layout.addWidget(self.image_label2)

        # Set up the capture buttons for the two players

        capture_btn_layout = QHBoxLayout()
        capture_btn_layout.addWidget(self.capture_btn1)

        # Set up the text label for the result of the game

        bottom_text_layout = QVBoxLayout()
        bottom_text_layout.addWidget(self.bottom_text)
        bottom_text_layout.addWidget(self.winner_text)

        # Add the constructed layouts to the right widget

        right_layout.addLayout(top_btn_layout)
        right_layout.addLayout(camera_layout)
        right_layout.addLayout(capture_btn_layout)
        right_layout.addLayout(bottom_text_layout)

        self.right_widget.setLayout(right_layout)

        # === Set up the main widget's layout (this will contain the left and right widgets respectively) ===

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_widget)
        main_layout.addWidget(self.right_widget)
        main_layout.setStretch(1, 200)

        self.main_widget.setLayout(main_layout)
        self.setCentralWidget(self.main_widget)

    def neural_network(self):
        """Hand signal detection using neural network."""
        self.player1_choice = homemade_nn.predict(self.cv_img)
        if self.difficulty == '1v1':
            self.player2_choice = homemade_nn.predict(self.cv_img2)
        else:
            if self.difficulty != 'normal':
                update_player1_chances(self.player1_choice) # Collect player patterns if not in normal mode
                self.player2_choice = self.ai_action(self.player1_choice)
            else:
                self.player2_choice = get_normal_action(self.player1_choice)
            self.update_ai_image()

        self.update_bottom_text()

    def media_pipe(self):
        """Hand signal detection using MediaPipe."""
        with ChangeDirectory('mediapipe_hand_detection'):
            self.player1_choice = rps_utils.detect_gesture(self.cv_img)
            if self.difficulty == '1v1':
                self.player2_choice = rps_utils.detect_gesture(self.cv_img2)
            else:
                if self.difficulty != 'normal':
                    update_player1_chances(self.player1_choice) # Collect player patterns if not normal mode
                    self.player2_choice = self.ai_action(self.player1_choice)
                else:
                    self.player2_choice = get_normal_action(self.player1_choice)
            self.update_ai_image()
            
        self.update_bottom_text()

    def capture_player1_image(self):
        if self.detection_method == 'neural network':
            self.neural_network()
        if self.detection_method == 'mediapipe':
            self.media_pipe()

        self.update_bottom_text()

    def capture_player2_image(self):
        if self.detection_method == 'neural network':
            self.neural_network()
        if self.detection_method == 'mediapipe':
            self.media_pipe()

        self.update_bottom_text()

    def switch_difficulty(self, difficulty='easy'):
        """Switch the difficulty of the game."""
        # TODO: switch self.ai_action according to difficulty
        self.difficulty = difficulty
        if difficulty == 'easy':
            self.ai_action = get_easy_action
        elif difficulty == 'normal':
            self.ai_action = get_normal_action
        elif difficulty == 'hard':
            self.ai_action = get_hard_action

        self.highlight_button((self.sidebar_btn1, self.sidebar_btn2, self.sidebar_btn3, self.sidebar_btn4), difficulty)
        self.thread.set_cropping_method('grab_side' if difficulty == '1v1' else 'grab_mid')
        self.bottom_text.setText('')

    def switch_detection_method(self, detection_method='neural network'):
        """Switch the detection method."""
        self.detection_method = detection_method
        self.highlight_button((self.neural_network_btn, self.media_pipe_btn), detection_method)
        self.bottom_text.setText('')

    def highlight_button(self, btn_tuple, btn_text):
        """Highlight a button with a given text from a tuple of buttons."""
        for btn in btn_tuple:
            if btn.text().lower() == btn_text.lower():
                btn.setStyleSheet('QPushButton { background-color: #000; color: #fff; }')
            else:
                btn.setStyleSheet('QPushButton { background-color: lightgray; color: initial; }')

    def update_bottom_text(self):
        """Display the recognized hand symbols as the bottom text."""
        self.bottom_text.setText(f'Player 1 chose {self.player1_choice.lower()}, Player 2 chose {self.player2_choice.lower()}')
        self.winner_text.setText(get_winner(self.player1_choice, self.player2_choice))

    def update_image(self, cv_img, cv_img2):
        """Update the camera image(s)."""
        self.cv_img = cv_img
        qt_img = self.convert_cv_qt_color(cv_img)
        self.image_label.setPixmap(qt_img)

        if self.difficulty == '1v1':
            self.cv_img2 = cv_img2
            qt_img = self.convert_cv_qt_color(cv_img2)
            self.image_label2.setPixmap(qt_img)
        else:
            self.image_label2.setPixmap(self.pixmap)

    def update_ai_image(self):
        image_file = f'{self.player2_choice}.jpg'
        image_filepath = f'sample_images/{image_file}'
        self.pixmap = QPixmap(image_filepath).scaled(self.img_width, self.img_height)
        self.image_label2.setPixmap(self.pixmap)

    def convert_cv_qt_color(self, cv_img):
        """ Convert from a BGR opencv image to QPixmap. """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.img_width, self.img_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def convert_cv_qt_gray(self, cv_img, opponent_image=False):
        """ Convert from a grayscale opencv image to QPixmap. """
        converted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        if not opponent_image:
            self.cv_img = converted
        else:
            self.cv_img2 = converted
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        h, w = rgb_image.shape
        bytes_per_line = w

        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        p = convert_to_qt_format.scaled(self.img_width, self.img_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)
