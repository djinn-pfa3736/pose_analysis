import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QSplashScreen, QSlider, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

import subprocess
import argparse
import pdb

class ComparisonExecuter(QtWidgets.QWidget):

    def __init__(self, file_name_1, file_name_2, threshold):
            
        super(ComparisonExecuter, self).__init__()
        # super().__init__()
        # self.ui = LaughMakerUI()
        # self.ui.setupUI(self)

        # self.flashSplash()

        self.prepare_data(file_name_1, file_name_2, threshold)
        self.create_window(file_name_1)

    def create_window(self, file_name):
        image = self.read_one_image(file_name)
        
        self.height, self.width, depth = image.shape
        
        window_width = self.width + 15
        window_height = self.height + 45
        self.resize(window_width, window_height)
                
        """
        self.leftView = QtWidgets.QGraphicsView(self)
        self.leftView.setGeometry(QtCore.QRect(5, 5, width, height))
        self.leftView.setObjectName('leftView')

        self.rightView = QtWidgets.QGraphicsView(self)
        self.rightView.setGeometry(QtCore.QRect(10 + width, 5, width, height))
        self.rightView.setObjectName('rightView')
        """
        
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.setGeometry(35 + int(self.width/3), 10 + int(self.height/3), -55 + int(self.width/3), 30)
        self.slider.valueChanged[int].connect(self.changeValue)
    
        self.info_start_x = 700
        self.info_start_y = 450
        self.info_joint_width = 100
        self.info_joint_height = 10
        self.info_diff_width = 50
        self.info_diff_height = 10
        self.info_margin_x = 10
        self.info_margin_y = 20
        self.info_separator_width = 50
    
        diff = np.abs(self.paired_angles_1[0] - self.paired_angles_2[0])
        diff_sort_args = np.argsort(diff)[::-1]
        
        self.labels = [0]*16
        self.diff = [0]*16
        for i in range(8):
            x = self.info_start_x
            y = self.info_start_y + i*(self.info_joint_height + self.info_margin_y)
            
            self.labels[i] = QLabel(self)
            self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
            self.labels[i].setGeometry(x, y, self.info_joint_width, self.info_joint_height)
            
            self.diff[i] = QLabel(self)
            self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))
            self.diff[i].setGeometry(x + self.info_joint_width + self.info_margin_x, y, self.info_diff_width, self.info_diff_height)


        for i in range(8,16):
            x = self.info_start_x + self.info_joint_width + self.info_diff_width + self.info_separator_width
            y = self.info_start_y + (i - 8)*(self.info_joint_height + self.info_margin_y)
            
            self.labels[i] = QLabel(self)
            self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
            self.labels[i].setGeometry(x, y, self.info_joint_width, self.info_joint_height)
            
            self.diff[i] = QLabel(self)
            self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))
            self.diff[i].setGeometry(x + self.info_joint_width + self.info_margin_x, y, self.info_diff_width, self.info_diff_height)
                    
        # pdb.set_trace()
        
        self.graphView = QtWidgets.QGraphicsView(self)
        self.graphView.setGeometry(5, 10 + int(self.height/3), int(self.width/3), int(self.height/3))
        self.graphView.setObjectName("graphView") 
        # self.graphView.setStyleSheet("background-color:transparent;border:transparent;")

        self.figure = Figure(figsize=(5, 3))
        """
        self.figure = Figure(figsize=(1.9, 1.4))    
        FigureCanvas.__init__(self, self.figure)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        """
        self.axes = self.figure.add_subplot(111)
        self.axes.tick_params(axis='x', labelsize=6)
        self.axes.tick_params(axis='y', labelsize=6)
        self.figure.patch.set_facecolor('white')
        # self.figure.patch.set_alpha(0.5)
        # self.axes.patch.set_alpha(0.0)
        self.figure.subplots_adjust(left=0.05, right=1.0, bottom=0.05, top=1.0)
        self.axes.grid(color='black', linestyle='dotted', linewidth=0.5)
        self.canvas = FigureCanvas(self.figure)        
        # self.canvas.setStyleSheet("background-color:transparent;border:transparent;")
    
        self.graph_scene = QtWidgets.QGraphicsScene()
        self.graph_scene.addWidget(self.canvas)
    
        self.graphView.setScene(self.graph_scene)
        # pdb.set_trace()
        self.axes.set_ylim(0, 360)
        self.line, = self.axes.plot(np.abs(self.paired_angles_1[0] - self.paired_angles_2[0]), linewidth = 0.7)
    
        image_rgb_1 = cv2.cvtColor(self.paired_images_1[0], cv2.COLOR_BGR2RGB)
        qimage_1 = QImage(image_rgb_1.flatten(), self.width, self.height, QImage.Format_RGB888)
        self.leftImageLabel = QLabel(self)
        pixmap = QPixmap.fromImage(qimage_1)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.leftImageLabel.setPixmap(pixmap)
        self.leftImageLabel.setGeometry(5, 5, int(self.width/3), int(self.height/3))
        self.leftImageLabel.scaleFactor = 1.0

        image_rgb_2 = cv2.cvtColor(self.paired_images_2[0], cv2.COLOR_BGR2RGB)
        qimage_2 = QImage(image_rgb_2.flatten(), self.width, self.height, QImage.Format_RGB888)
        self.rightImageLabel = QLabel(self)
        pixmap = QPixmap.fromImage(qimage_2)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.rightImageLabel.setPixmap(pixmap)
        self.rightImageLabel.setGeometry(10 + int(self.width/3), 5, int(self.width/3), int(self.height/3))
        self.rightImageLabel.scaleFactor = 1.0

        self.setWindowTitle('Motion Comparison Executer')
        self.show()
        self.canvas.draw()
        
    def changeValue(self, value):

        position = int((value/100)*len(self.paired_images_1))

        diff = np.abs(self.paired_angles_1[position] - self.paired_angles_2[position])
        diff_sort_args = np.argsort(diff)[::-1]
        
        for i in range(8):
            x = self.info_start_x
            y = self.info_start_y + i*(self.info_joint_height + self.info_margin_y)
            
            self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
            
            self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))

        for i in range(8,16):
            x = self.info_start_x + self.info_joint_width + self.info_diff_width + self.info_separator_width
            y = self.info_start_y + (i - 8)*(self.info_joint_height + self.info_margin_y)
            
            self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
            
            self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))
        
        image_rgb_1 = cv2.cvtColor(self.paired_images_1[position], cv2.COLOR_BGR2RGB)
        qimage_1 = QImage(image_rgb_1.flatten(), self.width, self.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage_1)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.leftImageLabel.setPixmap(pixmap)
        self.leftImageLabel.setGeometry(5, 5, int(self.width/3), int(self.height/3))

        image_rgb_2 = cv2.cvtColor(self.paired_images_2[position], cv2.COLOR_BGR2RGB)
        qimage_2 = QImage(image_rgb_2.flatten(), self.width, self.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage_2)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.rightImageLabel.setPixmap(pixmap)
        self.rightImageLabel.setGeometry(10 + int(self.width/3), 5, int(self.width/3), int(self.height/3))
    
        self.line.set_ydata(np.abs(self.paired_angles_1[position] - self.paired_angles_2[position]))
        self.canvas.draw()
        
    def read_one_image(self, file_name):
        cap = cv2.VideoCapture(file_name)
        success, image = cap.read()
        
        return image
        
    def calc_inner(self, a, b):
        a = np.array(a)
        b = np.array(b)
        inner = np.sum(a*b)
        
        return inner
        
    def calc_angle(self, x1, x2, x3):
        x1 = np.array(x1)
        x2 = np.array(x2)
        x3 = np.array(x3)
        
        a = x1 - x2
        b = x3 - x2
        
        inner = self.calc_inner(a, b)
        cos_theta = inner/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))
        theta = np.rad2deg(np.arccos(cos_theta))
        
        return theta

    def convert_to_vector(self, landmark):
        x = np.array([landmark.x, landmark.y, landmark.z])
        
        return x

    def calc_diff(self, angle_tables):
        diff_angle_tables = []
        for i in range(len(angle_tables) - 1):
            diff_angle_tables.append(angle_tables[i + 1] - angle_tables[i])

        diff_angle_tables = np.array(diff_angle_tables)
        
        return diff_angle_tables

    def calc_angle_from_indices(self, landmarks, id1, id2, id3):
        x1 = self.convert_to_vector(landmarks.landmark[id1])
        x2 = self.convert_to_vector(landmarks.landmark[id2])
        x3 = self.convert_to_vector(landmarks.landmark[id3])

        theta = self.calc_angle(x1, x2, x3)
        
        return theta

    def extract_angle_tables(self, file_name, pose):

        cap = cv2.VideoCapture(file_name)
            
        angle_tables = []
        angle_diff_tables = []
        frame_tables =  []
        image_tables = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue
                break
                
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True

            angle_table = []
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 11, 12, 14))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 12, 11, 13))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 12, 14, 16))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 14, 12, 24))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 11, 12, 24))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 12, 11, 23))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 13, 11, 23))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 11, 13, 15))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 12, 24, 26))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 12, 24, 23))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 11, 23, 24))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 11, 13, 25))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 26, 24, 23))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 25, 23, 24))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 24, 26, 28))
            angle_table.append(self.calc_angle_from_indices(results.pose_landmarks, 23, 25, 27))

            angle_tables.append(angle_table)
            frame_tables.append(results.pose_landmarks)
            image_tables.append(image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

        angle_tables = np.array(angle_tables)
        angle_diff_tables = self.calc_diff(angle_tables)
        cap.release()
        
        return angle_tables, angle_diff_tables, frame_tables, image_tables

    def calc_distance_between_frames(self, frame1, frame2):
        dist = np.sqrt(np.sum((frame1 - frame2)**2))/len(frame1)

        return dist

    def calc_corr_between_frames(self, frame1, frame2):
        frame1 = np.array(frame1)
        frame2 = np.array(frame2)
        corr = np.corrcoef(frame1, frame2)
        
        return corr[0, 1]

    def align_frames(self, tables1, tables2, match_threshold):

        dist_mat = np.zeros((len(tables1), len(tables2)))
        for i in range(len(tables1)):
            for j in range(len(tables2)):
                dist_mat[i, j] = self.calc_distance_between_frames(tables1[i], tables2[j])

        dp_mat = np.zeros((len(tables1) + 1, len(tables2) + 1))
        
        dir_mat = np.zeros((len(tables1) + 1, len(tables2) + 1))
        for i in range(1, len(tables1) + 1):
            dir_mat[i, 0] = 2
        
        for i in range(1, len(tables1) + 1):
            for j in range(1, len(tables2) + 1):
                if dist_mat[i - 1, j - 1] <= match_threshold:
                    match_val = 1
                else:
                    match_val = -1
                
                val1 = dp_mat[i, j - 1]    
                val2 = dp_mat[i - 1, j - 1] + match_val
                val3 = dp_mat[i - 1, j]
                
                max_val = np.max((val1, val2, val3))
                dp_mat[i, j] = max_val
                
                max_idx = np.argmax((val1, val2, val3))
                dir_mat[i, j] = max_idx

        pairs = []
        i = len(tables1)
        j = len(tables2)
        while i != 0 or j != 0:
            if dir_mat[i, j] == 0:
                pairs.append(['-', j - 1])
                j -= 1
            elif dir_mat[i, j] == 1:
                pairs.append([i - 1, j - 1])
                i -= 1
                j -= 1
            else:
                pairs.append([i - 1, '-'])
                i -= 1
                
        pairs.reverse()
        return pairs, dist_mat

    def align_frames_with_corr(self, tables1, tables2, match_threshold):

        corr_mat = np.zeros((len(tables1), len(tables2)))
        for i in range(len(tables1)):
            for j in range(len(tables2)):
                corr_mat[i, j] = self.calc_corr_between_frames(tables1[i], tables2[j])
        # pdb.set_trace()
        
        dp_mat = np.zeros((len(tables1) + 1, len(tables2) + 1))
        
        dir_mat = np.zeros((len(tables1) + 1, len(tables2) + 1))
        for i in range(1, len(tables1) + 1):
            dir_mat[i, 0] = 2
        
        for i in range(1, len(tables1) + 1):
            for j in range(1, len(tables2) + 1):
                if corr_mat[i - 1, j - 1] >= match_threshold:
                    match_val = 1
                else:
                    match_val = -1
                
                val1 = dp_mat[i, j - 1]    
                val2 = dp_mat[i - 1, j - 1] + match_val
                val3 = dp_mat[i - 1, j]
                
                max_val = np.max((val1, val2, val3))
                dp_mat[i, j] = max_val
                
                max_idx = np.argmax((val1, val2, val3))
                dir_mat[i, j] = max_idx

        pairs = []
        i = len(tables1)
        j = len(tables2)
        while i != 0 or j != 0:
            if dir_mat[i, j] == 0:
                pairs.append(['-', j - 1])
                j -= 1
            elif dir_mat[i, j] == 1:
                pairs.append([i - 1, j - 1])
                i -= 1
                j -= 1
            else:
                pairs.append([i - 1, '-'])
                i -= 1
                
        pairs.reverse()
        return pairs, corr_mat

    def prepare_data(self, file1, file2, threshold):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.joint_info = []
        self.joint_info = ['14-12-11', '13-11-12', '12-14-16', '14-12-24', '24-12-11', '23-11-12']
        self.joint_info += ['13-11-23', '11-13-15', '12-24-26', '12-24-23', '11-23-24', '11-23-25']
        self.joint_info += ['26-24-23', '25-23-24', '24-26-28', '23-25-27']
        
        print('*** Calculation of angles has been started!! ***')
        self.angle_tables_1, self.angle_diff_tables_1, self.frame_tables_1, self.image_tables_1 = self.extract_angle_tables(args.file1, pose)
        self.angle_tables_2, self.angle_diff_tables_2, self.frame_tables_2, self.image_tables_2 = self.extract_angle_tables(args.file2, pose)

        print('*** Alignment of frames has been started!! ***')
        """
        pairs, dist_mat = align_frames(angle_tables_1, angle_tables_2, np.pi/18)
        pair_dist_vec = []
        for pair in pairs:
            if (pair[0] != '-') and (pair[1] != '-'):
                pair_dist_vec.append(dist_mat[pair[0], pair[1]])
        print(str(np.mean(pair_dist_vec)) + ' ± ' + str(np.sqrt(np.var(pair_dist_vec))))
        print(str(np.min(pair_dist_vec)) + ' - ' + str(np.max(pair_dist_vec)))
        """

        self.pairs, self.corr_mat = self.align_frames_with_corr(self.angle_diff_tables_1, self.angle_diff_tables_2, threshold)
        self.pair_corr_vec = []
        
        # pdb.set_trace()
        
        # file_count = 0
        
        self.paired_images_1 = []
        self.paired_images_2 = []
        self.paired_angles_1 = []
        self.paired_angles_2 = []
        self.pair_matched_indices = []
        
        """
        for i in range(len(self.pairs)):
            pair = self.pairs[i]
            
            if (pair[0] != '-') and (pair[1] != '-'):
                self.pair_matched_indices.append(i)
                
                image1 = cv2.cvtColor(self.image_tables_1[pair[0]], cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image1,
                    self.frame_tables_1[pair[0]],
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())        
                self.paired_images_1.append(image1)

                image2 = cv2.cvtColor(self.image_tables_2[pair[1]], cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image2,
                    self.frame_tables_2[pair[1]],
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                self.paired_images_2.append(image2)
                
                self.pair_corr_vec.append(self.corr_mat[pair[0], pair[1]])
                
                # cv2.imwrite('compare_{:0>4}'.format(file_count) + '.png', image)
                # file_count += 1
        """
        
        pair = self.pairs[0]
        
        first_frame_set_flag_1 = False
        first_frame_set_flag_2 = False
        first_gap_len_1 = 0
        first_gap_len_2 = 0
        previous_image_1 = pair[0]
        previous_image_2 = pair[1]
        previous_angle_1 = []
        previous_angle_2 = []
        for i in range(len(self.pairs)):
            pair = self.pairs[i]
            
            if pair[0] != '-':
                
                image1 = cv2.cvtColor(self.image_tables_1[pair[0]], cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image1,
                    self.frame_tables_1[pair[0]],
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  
                
                if not first_frame_set_flag_1:
                    for i in range(first_gap_len_1):
                        self.paired_images_1 = [image1] + self.paired_images_1
                        self.paired_angles_1 = [self.angle_tables_1[pair[0]]] + self.paired_angles_1
                    first_frame_set_flag_1 = True
                else:      
                    self.paired_images_1.append(image1)
                    self.paired_angles_1.append(self.angle_tables_1[pair[0]])
                previous_image_1 = image1
                previous_angle_1 = self.angle_tables_1[pair[0]]
            
            else:
                if not first_frame_set_flag_1:
                    first_gap_len_1 += 1
                else:
                    self.paired_images_1.append(previous_image_1)
                    self.paired_angles_1.append(previous_angle_1)
                        
            if pair[1] != '-':
                
                image2 = cv2.cvtColor(self.image_tables_2[pair[1]], cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image2,
                    self.frame_tables_2[pair[1]],
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  
                
                if not first_frame_set_flag_2:
                    for i in range(first_gap_len_2):
                        self.paired_images_2 = [image2] + self.paired_images_2
                        self.paired_angles_2 = [self.angle_tables_2[pair[1]]] + self.paired_angles_2
                    first_frame_set_flag_2 = True
                else:      
                    self.paired_images_2.append(image2)
                    self.paired_angles_2.append(self.angle_tables_2[pair[1]])
                previous_image_2 = image2
                previous_angle_2 = self.angle_tables_2[pair[1]]
            
            else:
                if not first_frame_set_flag_2:
                    first_gap_len_2 += 1
                else:
                    self.paired_images_2.append(previous_image_2)
                    self.paired_angles_2.append(previous_angle_2)

    def check_matching_count(self, paires):
        matching_count = 0
        
        for pair in paires:
            if (pair[0] != '-') and (pair[1] != '-'):
                matching_count += 1
        
        return matching_count
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='file1.MOV')
    parser.add_argument('--file2', type=str, default='file2.MOV')
    parser.add_argument('--threshold', type=float, default=0.4)
    # parser.add_argument('--output', type=str, default='compare.mp4')
    args = parser.parse_args()

    app = QApplication([])
    comparison_executer = ComparisonExecuter(args.file1, args.file2, args.threshold)
    sys.exit(app.exec_())


    # res = subprocess.call('bash ./create_movie.sh ' + args.output, shell=True)
