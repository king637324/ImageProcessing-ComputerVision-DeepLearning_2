from sys import int_info
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from time import sleep

import hw2_ui as ui

import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np
import glob
import imutils
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA


class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        ####################
        #####    Q1    #####
        #################### 
        self.grid = (11, 8)
        self.ButtonQ1_1.clicked.connect(self.Q1_1)
        self.ButtonQ1_2.clicked.connect(self.Q1_2)

        ####################
        #####    Q2    #####
        #################### 
        self.ButtonQ2_1.clicked.connect(self.Q2_1)
        self.ButtonQ2_2.clicked.connect(self.Q2_2)
        self.ButtonQ2_3.clicked.connect(self.Q2_3)
        choices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        self.comboBox_Q2_3.addItems(choices)
        self.ButtonQ2_4.clicked.connect(self.Q2_4)
        self.ButtonQ2_5.clicked.connect(self.Q2_5)
        self.intrinsic_matrix = 0
        self.distortion = 0
        self.rotation_vectors = 0
        self.translation_vectors = 0
        self.Q2()

        ####################
        #####    Q3    #####
        #################### 
        self.ButtonQ3_1.clicked.connect(self.Q3_1)
        self.ButtonQ3_2.clicked.connect(self.Q3_2)
        ####################
        #####    Q4    #####
        #################### 
        self.ButtonQ4_1.clicked.connect(self.Q4_1)


    def Q1_1(self):
        print("-------------Q1_1-------------")
        
        print("-------------Q1_1 Finsh-------------\n")

    def Q1_2(self):
        print("-------------Q1_2-------------")
        
        print("-------------Q1_2 Finsh-------------\n")

    def Q2(self):
        # glob.glob可以同時獲得這個路徑的所有的匹配文件
        picture = glob.glob('Dataset_OpenCvDl_Hw2/Q2_Image/*.bmp')
        # 設定終止條件，迭代30次或移動0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 儲存圖像點
        objpoints = [] # 3維空間
        picture_points = [] # 2維空間
        
        # 處理每張照片
        for filename in picture:
            # object point 初始化宣告 3表示RBG三個圖片
            objp = np.zeros((self.grid[0] * self.grid[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.grid[0],0:self.grid[1]].T.reshape(-1,2)
            
            # 讀取這個檔名的圖片
            pic = cv2.imread(filename)
            # 將圖片換成灰階
            gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)

            # 找 棋盤的corners   cv2.findChessboardCorners(圖片, (col, row),None)
            # col跟row 是棋盤的線
            ret, corners = cv2.findChessboardCorners(gray, self.grid,None)

            # 如果有找到corner 就把找到的點畫出來
            if ret == True:
                # cornerSubPix(轉灰階的圖片,找到的corners,(col,row),)
                corners2 = cv2.cornerSubPix(gray,corners, self.grid,(-1,-1),criteria)

                # 將計算與找到的圖像點儲存進陣列
                objpoints.append(objp)
                picture_points.append(corners2)

        ret, intrinsic_matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(objpoints, picture_points, gray.shape[::-1],None,None)

        self.intrinsic_matrix = intrinsic_matrix
        self.distortion = distortion
        self.rotation_vectors = rotation_vectors
        self.translation_vectors = translation_vectors

        
    def Q2_1(self):
        print("-------------Q2_1-------------")

        # glob.glob可以同時獲得這個路徑的所有的匹配文件
        picture = glob.glob('Dataset_OpenCvDl_Hw2/Q2_Image/*.bmp')
        # 設定終止條件，迭代30次或移動0.001
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 處理每張照片
        for filename in picture:
            # 讀取這個檔名的圖片
            pic = cv2.imread(filename)
            pic = cv2.resize(pic, (480, 480))
            # 將圖片換成灰階
            gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)

            # 找 棋盤的corners   cv2.findChessboardCorners(圖片, (col, row),None)
            # col跟row 是棋盤的線
            ret, corners = cv2.findChessboardCorners(gray, self.grid,None)

            # 如果有找到corner 就把找到的點畫出來
            if ret == True:
                # cornerSubPix(轉灰階的圖片,找到的corners,(col,row),)
                corners2 = cv2.cornerSubPix(gray,corners, self.grid,(-1,-1),criteria)

                # 畫出找到的corners
                pic = cv2.drawChessboardCorners(pic, self.grid, corners2,ret)
                print(filename)
                cv2.imshow('Q2_1',pic)
                # 顯示500ms
                cv2.waitKey(500)
        # 關掉這張照片
        cv2.destroyAllWindows()
        
        print("-------------Q2_1 Finsh-------------\n")

    def Q2_2(self):
        print("-------------Q2_2-------------")

        print("Intrinsic Matrix：")
        print(self.intrinsic_matrix,"\n")
        
        print("-------------Q2_2 Finsh-------------\n")
    
    def Q2_3(self):
        print("-------------Q2_3-------------")

        # 將下拉選單選到的數字轉乘int
        picture_number = int(self.comboBox_Q2_3.currentText())
        print("you choice picture",picture_number)

        rotation_matrix,_ = cv2.Rodrigues(self.rotation_vectors[picture_number-1])
        Extrinsic_Matrix = np.append(rotation_matrix, self.translation_vectors[picture_number-1],axis=1)
        
        print("Extrinsic Matrix：")
        print(Extrinsic_Matrix,"\n")
        
        print("-------------Q2_3 Finsh-------------\n")
    
    def Q2_4(self):
        print("-------------Q2_4-------------")

        print("Distortion Matrix：")
        print(self.distortion,"\n")
        
        print("-------------Q2_4 Finsh-------------\n")

    def Q2_5(self):
        print("-------------Q2_5-------------")

        # glob.glob可以同時獲得這個路徑的所有的匹配文件
        picture = glob.glob('Dataset_OpenCvDl_Hw2/Q2_Image/*.bmp')

        # 處理每張照片
        for filename in picture:
            # object point 初始化宣告 3表示RBG三個圖片
            objp = np.zeros((self.grid[0] * self.grid[1],3), np.float32)
            objp[:,:2] = np.mgrid[0:self.grid[0],0:self.grid[1]].T.reshape(-1,2)

            
            
            # 讀取這個檔名的圖片
            pic = cv2.imread(filename)
            # 重新設定圖片大小
            pic = cv2.resize(pic, (480, 480))
            u, v = pic.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.intrinsic_matrix, self.distortion, (u, v), 0, (u, v))

            # 纠正畸变
            dst = cv2.undistort(pic, self.intrinsic_matrix, self.distortion,None,newcameramtx)

            # 把兩張照片弄成一個視窗
            display = np.hstack([pic,dst])
            print(filename)
            cv2.imshow('Q2_5',display)

            # 顯示500ms
            cv2.waitKey(500)
        # 關掉這張照片
        cv2.destroyAllWindows()
        
        print("-------------Q2_5 Finsh-------------\n")
    
    def Q3_1(self):
        print("-------------Q3_1-------------")

        print("-------------Q3_1 Finsh-------------\n")

    def Q3_2(self):
        print("-------------Q3_2-------------")

        print("-------------Q3_2 Finsh-------------\n")
    
    def Q4_1(self):
        print("-------------Q4_1-------------")

        print("-------------Q4_1 Finsh-------------\n")
       


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())