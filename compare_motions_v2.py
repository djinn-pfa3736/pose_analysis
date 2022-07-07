import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QSplashScreen, QSlider, QLabel, QApplication, QCheckBox, QPushButton
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

        self.position_slider1 = np.arange(1)
        self.position_slider2 = np.arange(1)
        self.slider_value1 = 0
        self.slider_value2 = 0

        self.prepare_data(file_name_1, file_name_2)
        self.create_window(file_name_1)

    def create_window(self, file_name):

        image = self.read_one_image(file_name)
        
        self.height, self.width, depth = image.shape
        
        window_width = self.width + 15
        window_height = self.height + 45
        self.resize(window_width, window_height)
        
        self.image_label_1 = QLabel(self)
        self.image_label_1.setText("生徒")
        self.image_label_1.setGeometry(5, 5, 100, 15)

        self.image_label_2 = QLabel(self)
        self.image_label_2.setText("教師")
        self.image_label_2.setGeometry(5 + int(self.width/3), 5, 100, 15)

        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider1.setFocusPolicy(Qt.NoFocus)
        self.slider1.setGeometry(35, 30 + int(self.height/3), -55 + int(self.width/3), 30)
        self.slider1.valueChanged[int].connect(self.change_value_slider1)

        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setFocusPolicy(Qt.NoFocus)
        self.slider2.setGeometry(35 + int(self.width/3), 30 + int(self.height/3), -55 + int(self.width/3), 30)
        self.slider2.valueChanged[int].connect(self.change_value_slider2)

        self.info_start_x = 700
        self.info_start_y = 450
        self.info_joint_width = 100
        self.info_joint_height = 20
        self.info_diff_width = 50
        self.info_diff_height = 20
        self.info_margin_x = 10
        self.info_margin_y = 20
        self.info_separator_width = 50
    
        diff = np.abs(self.angle_tables_1[0] - self.angle_tables_2[0])
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
        
        self.check_box_list = []
        check_box1 = QCheckBox(self.joint_info[0], self)
        check_box2 = QCheckBox(self.joint_info[1], self)
        check_box3 = QCheckBox(self.joint_info[2], self)
        check_box4 = QCheckBox(self.joint_info[3], self)
        check_box5 = QCheckBox(self.joint_info[4], self)
        check_box6 = QCheckBox(self.joint_info[5], self)
        check_box7 = QCheckBox(self.joint_info[6], self)
        check_box8 = QCheckBox(self.joint_info[7], self)
        check_box9 = QCheckBox(self.joint_info[8], self)
        check_box10 = QCheckBox(self.joint_info[9], self)
        check_box11 = QCheckBox(self.joint_info[10], self)
        check_box12 = QCheckBox(self.joint_info[11], self)
        check_box13 = QCheckBox(self.joint_info[12], self)
        check_box14 = QCheckBox(self.joint_info[13], self)
        check_box15 = QCheckBox(self.joint_info[14], self)
        check_box16 = QCheckBox(self.joint_info[15], self)

        check_box1.setChecked(True)
        check_box2.setChecked(True)
        check_box3.setChecked(True)
        check_box4.setChecked(True)
        check_box5.setChecked(True)
        check_box6.setChecked(True)
        check_box7.setChecked(True)
        check_box8.setChecked(True)
        check_box9.setChecked(True)
        check_box10.setChecked(True)
        check_box11.setChecked(True)
        check_box12.setChecked(True)
        check_box13.setChecked(True)
        check_box14.setChecked(True)
        check_box15.setChecked(True)
        check_box16.setChecked(True)

        self.check_box_list.append(check_box1)
        self.check_box_list.append(check_box2)
        self.check_box_list.append(check_box3)
        self.check_box_list.append(check_box4)
        self.check_box_list.append(check_box5)
        self.check_box_list.append(check_box6)
        self.check_box_list.append(check_box7)
        self.check_box_list.append(check_box8)
        self.check_box_list.append(check_box9)
        self.check_box_list.append(check_box10)
        self.check_box_list.append(check_box11)
        self.check_box_list.append(check_box12)
        self.check_box_list.append(check_box13)
        self.check_box_list.append(check_box14)
        self.check_box_list.append(check_box15)
        self.check_box_list.append(check_box16)

        check_box_x = 20 + int(self.width/3)*2
        check_box_y = 20
        check_box_width = 100
        check_box_height = 20
        check_box_margin = 20

        for i in range(8):
            self.check_box_list[i].setGeometry(check_box_x, check_box_y + i*check_box_height, check_box_width, check_box_height)

        for i in range(8, 16):
            self.check_box_list[i].setGeometry(check_box_x + check_box_width + check_box_margin, check_box_y + (i - 8)*check_box_height, check_box_width, check_box_height)

        self.graphView = QtWidgets.QGraphicsView(self)
        self.graphView.setGeometry(5, 70 + int(self.height/3), int(self.width/3), int(self.height/3))
        self.graphView.setObjectName("graphView") 
        # self.graphView.setStyleSheet("background-color:transparent;border:transparent;")

        button_width = 100
        button_height = 20
        # self.apply_button = QPushButton("適用", self)
        # self.apply_button.setGeometry(check_box_x + check_box_width + check_box_margin, check_box_y + 8*check_box_height, button_width, button_height)
        # self.apply_button.clicked.connect(self.apply_clicked)

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
        self.axes.set_ylim(0, np.max(np.abs(self.angle_tables_1[0] - self.angle_tables_2[0])) + 20)
        self.line, = self.axes.plot(np.abs(self.angle_tables_1[0] - self.angle_tables_2[0]), linewidth = 0.7)
    
        image_bgr_1 = cv2.cvtColor(self.image_tables_1[self.slider_value1], cv2.COLOR_BGR2RGB)
        self.mp_drawing.draw_landmarks(
            image_bgr_1,
            self.frame_tables_1[self.slider_value1],
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())           
        qimage_1 = QImage(image_bgr_1.flatten(), self.width, self.height, QImage.Format_RGB888)
        self.leftImageLabel = QLabel(self)
        pixmap = QPixmap.fromImage(qimage_1)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.leftImageLabel.setPixmap(pixmap)
        self.leftImageLabel.setGeometry(5, 25, int(self.width/3), int(self.height/3))
        self.leftImageLabel.scaleFactor = 1.0

        image_bgr_2 = cv2.cvtColor(self.image_tables_2[self.slider_value2], cv2.COLOR_BGR2RGB)
        self.mp_drawing.draw_landmarks(
            image_bgr_2,
            self.frame_tables_2[self.slider_value2],
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        qimage_2 = QImage(image_bgr_2.flatten(), self.width, self.height, QImage.Format_RGB888)
        self.rightImageLabel = QLabel(self)
        pixmap = QPixmap.fromImage(qimage_2)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.rightImageLabel.setPixmap(pixmap)
        self.rightImageLabel.setGeometry(10 + int(self.width/3), 25, int(self.width/3), int(self.height/3))
        self.rightImageLabel.scaleFactor = 1.0

        self.setWindowTitle('動作比較プログラムメインウィンドウ')
        self.show()
        self.canvas.draw()
    
    def apply_clicked(self):
        print('test')

    def change_value_slider1(self, value):

        self.slider_value1 = value
        self.position_slider1 = int((value/100)*len(self.image_tables_1))
        # self.position_slider1 = np.arange(0, self.slider_value1)/100*len(self.image_tables_1)
        # self.position_slider1 = np.array([int(v) for v in self.position_slider1])

        position_image = int((value/100)*len(self.image_tables_1))

        # if len(self.position_slider1) == 0:
        #     self.position_slider1 = [0]
        # print(self.position_slider1)
        # print(self.position_slider2)
        diff = np.abs(self.angle_tables_1[self.position_slider1] - self.angle_tables_2[self.position_slider2])
        cum_diff = np.array([0.0]*len(diff[0]))
        plot_diff = cum_diff
        for i in range(len(diff)):
            cum_diff += diff[i]
        # diff_sort_args = np.argsort(diff)[::-1]

        for i in range(8):
            x = self.info_start_x
            y = self.info_start_y + i*(self.info_joint_height + self.info_margin_y)
            self.labels[i].setText('{:8s}'.format(self.joint_info[i]))

            if self.check_box_list[i].isChecked():
                
                # self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
                # self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))
                # print(cum_diff)
                self.diff[i].setText('{:5f}'.format(cum_diff[i]))
                plot_diff[i] = cum_diff[i]
            else:
                self.diff[i].setText('{:5f}'.format(-1.0))
                plot_diff[i] = -1

        for i in range(8,16):
            x = self.info_start_x + self.info_joint_width + self.info_diff_width + self.info_separator_width
            y = self.info_start_y + (i - 8)*(self.info_joint_height + self.info_margin_y)
            self.labels[i].setText('{:8s}'.format(self.joint_info[i]))

            # self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))    
            # self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))
        
            if self.check_box_list[i].isChecked():
                
                # self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
                # self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))

                self.diff[i].setText('{:5f}'.format(cum_diff[i]))
                plot_diff[i] = cum_diff[i]
            else:
                self.diff[i].setText('{:5f}'.format(-1.0))
                plot_diff[i] = -1

        image_bgr_1 = cv2.cvtColor(self.image_tables_1[position_image], cv2.COLOR_BGR2RGB)
        self.mp_drawing.draw_landmarks(
            image_bgr_1,
            self.frame_tables_1[position_image],
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()) 
        qimage_1 = QImage(image_bgr_1.flatten(), self.width, self.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage_1)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.leftImageLabel.setPixmap(pixmap)
        # self.leftImageLabel.setGeometry(5, 5, int(self.width/3), int(self.height/3))
        
        # self.line.set_ydata(np.abs(self.angle_tables_1[self.position_slider1] - self.angle_tables_2[self.position_slider2]))
        self.axes.set_ylim(0, np.max(plot_diff) + 20)
        self.line.set_ydata(np.abs(plot_diff))

        self.canvas.draw()

    def change_value_slider2(self, value):

        self.slider_value2 = value
        self.position_slider1 = int((value/100)*len(self.image_tables_1))
        # self.position_slider2 = np.arange(0, self.slider_value2)/100*len(self.image_tables_2)
        # self.position_slider2 = np.array([int(v) for v in self.position_slider2])

        position_image = int((value/100)*len(self.image_tables_2))

        # if len(self.position_slider2) == 0:
        #     self.position_slider2 = [0]
        # print(self.position_slider1)
        # print(self.position_slider2)
        diff = np.abs(self.angle_tables_1[self.position_slider1] - self.angle_tables_2[self.position_slider2])
        cum_diff = np.array([0.0]*len(diff[0]))
        plot_diff = cum_diff
        for i in range(len(diff)):
            cum_diff += diff[i]
        # diff_sort_args = np.argsort(diff)[::-1]

        for i in range(8):
            x = self.info_start_x
            y = self.info_start_y + i*(self.info_joint_height + self.info_margin_y)
            self.labels[i].setText('{:8s}'.format(self.joint_info[i]))

            if self.check_box_list[i].isChecked():
                
                # self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
                # self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))
                # print(cum_diff)
                self.diff[i].setText('{:5f}'.format(cum_diff[i]))
                plot_diff[i] = cum_diff[i]
            else:
                self.diff[i].setText('{:5f}'.format(-1.0))
                plot_diff[i] = -1

        for i in range(8,16):
            x = self.info_start_x + self.info_joint_width + self.info_diff_width + self.info_separator_width
            y = self.info_start_y + (i - 8)*(self.info_joint_height + self.info_margin_y)
            self.labels[i].setText('{:8s}'.format(self.joint_info[i]))

            # self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))    
            # self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))
        
            if self.check_box_list[i].isChecked():
                
                # self.labels[i].setText('{:8s}'.format(self.joint_info[diff_sort_args[i]]))
                # self.diff[i].setText('{:5f}'.format(diff[diff_sort_args[i]]))

                self.diff[i].setText('{:5f}'.format(cum_diff[i]))
                plot_diff[i] = cum_diff[i]
            else:
                self.diff[i].setText('{:5f}'.format(-1.0))
                plot_diff[i] = -1

        image_bgr_2 = cv2.cvtColor(self.image_tables_2[position_image], cv2.COLOR_BGR2RGB)
        self.mp_drawing.draw_landmarks(
            image_bgr_2,
            self.frame_tables_2[position_image],
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()) 
        qimage_2 = QImage(image_bgr_2.flatten(), self.width, self.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage_2)
        pixmap = pixmap.scaled(int(self.width/3), int(self.height/3), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.rightImageLabel.setPixmap(pixmap)
        # self.leftImageLabel.setGeometry(5, 5, int(self.width/3), int(self.height/3))
        
        # self.line.set_ydata(np.abs(self.angle_tables_1[self.position_slider1] - self.angle_tables_2[self.position_slider2]))
        self.axes.set_ylim(0, np.max(plot_diff) + 20)
        self.line.set_ydata(np.abs(plot_diff))

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
                # print("Ignoring empty camera frame.")
                print("Movie file has been loaded!!")
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

    def prepare_data(self, file1, file2):

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.joint_info = []
        # self.joint_info = ['14-12-11', '13-11-12', '12-14-16', '14-12-24', '24-12-11', '23-11-12']
        # self.joint_info += ['13-11-23', '11-13-15', '12-24-26', '12-24-23', '11-23-24', '11-23-25']
        # self.joint_info += ['26-24-23', '25-23-24', '24-26-28', '23-25-27']
        
        self.joint_info = ['左肩開き角', '右肩開き角', '左肘曲げ角', '左脇開き角', '左肩歪み角', '右肩歪み角']
        self.joint_info += ['右脇開き角', '右肘曲げ角', '左腰折り角', '左腰歪み角', '右腰歪み角', '右腰折り角']
        self.joint_info += ['左腿開き角', '右腿開き角', '左膝曲げ角', '右膝曲げ角']
        print('*** Calculation of angles has been started!! ***')
        self.angle_tables_1, self.angle_diff_tables_1, self.frame_tables_1, self.image_tables_1 = self.extract_angle_tables(file1, pose)
        self.angle_tables_2, self.angle_diff_tables_2, self.frame_tables_2, self.image_tables_2 = self.extract_angle_tables(file2, pose)


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
