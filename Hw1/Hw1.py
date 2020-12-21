from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import cv2
import numpy as np
from scipy import ndimage
import gui

class MainUi(QtWidgets.QWidget, gui.Ui_Form):
    def __init__(self, parent = None):
        super(MainUi, self).__init__(parent)
        self.setupUi(self)

        #connect function
        self.btnLoadImg.clicked.connect(self.btnLoadImgClicked)
        self.btnColorSep.clicked.connect(self.btnColorSepClicked)
        self.btnImgFlip.clicked.connect(self.btnImgFlipClicked)
        self.btnBlend.clicked.connect(self.btnBlendClicked)
        self.btnMedian.clicked.connect(self.btnMedianClicked)
        self.btnGaussian.clicked.connect(self.btnGaussianClicked)
        self.btnBilateral.clicked.connect(self.btnBilateralClicked)
        self.btnGaussianFilter.clicked.connect(self.btnGaussianFilterClicked)
        self.btnSobelX.clicked.connect(self.btnSobelXClicked)
        self.btnSobelY.clicked.connect(self.btnSobelYClicked)
        self.btnMagnitude.clicked.connect(self.btnMagnitudeClicked)
        self.btnTransform.clicked.connect(self.transform)

        #preprocess for part 3
        self.img3 = cv2.imread('./Q3_Image/Chihiro.jpg')
        self.img3_gray = cv2.cvtColor(self.img3, cv2.COLOR_BGR2GRAY)
        y, x = np.mgrid[-1:2, -1:2]
        gaussianKernel = np.exp(-(x ** 2 + y ** 2))
        gaussianKernel /= gaussianKernel.sum()

        dst = ndimage.convolve(self.img3_gray, gaussianKernel)
        dst = dst.astype(np.uint8)
        self.img3_gau = dst

        self.img_sx = np.zeros(self.img3.shape[:2])
        self.img_sy = np.zeros(self.img3.shape[:2])
        rows, cols = self.img3_gau.shape
        mSobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        mSobelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        for i in range(rows - 2):
            for j in range(cols - 2):
                self.img_sx[i + 1, j + 1] = abs(np.sum(self.img3_gau[i:i + 3, j:j + 3] * mSobelX))
                self.img_sy[i + 1, j + 1] = abs(np.sum(self.img3_gau[i:i + 3, j:j + 3] * mSobelY))

        self.img_sx = self.img_sx.astype(np.uint8)
        self.img_sy = self.img_sy.astype(np.uint8)

    #part 1
    def btnLoadImgClicked(self):
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        cv2.imshow('img', img)
        print('Height = ', img.shape[0])
        print('Width = ', img.shape[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnColorSepClicked(self):
        img = cv2.imread('./Q1_Image/Flower.jpg')
        imgb = np.zeros(img.shape, img.dtype)
        imgg = np.zeros(img.shape, img.dtype)
        imgr = np.zeros(img.shape, img.dtype)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                imgb[i, j] = [img[i, j, 0], 0, 0]
                imgg[i, j] = [0, img[i, j, 1], 0]
                imgr[i, j] = [0, 0, img[i, j, 2]]
        cv2.imshow('original', img)
        cv2.imshow('B', imgb)
        cv2.imshow('G', imgg)
        cv2.imshow('R', imgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnImgFlipClicked(self):
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        cv2.imshow('original', img)
        img = cv2.flip(img, 1)
        cv2.imshow('result', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnBlendClicked(self):
        def trackBarUpdate(x):
            a = cv2.getTrackbarPos('Blend', 'Blending') / 255
            imgn = cv2.addWeighted(img, a, imgf, (1 - a), 0.0)
            cv2.imshow('Blending', imgn)

        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        imgf = cv2.flip(img, 1)
        cv2.namedWindow('Blending')
        cv2.imshow('Blending', img)
        cv2.createTrackbar('Blend', 'Blending', 0, 255, trackBarUpdate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #part 2
    def btnMedianClicked(self):
        img = cv2.imread('./Q2_Image/Cat.png')
        cv2.imshow('original', img)
        imgn = cv2.medianBlur(img, 7)
        cv2.imshow('Blured', imgn)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnGaussianClicked(self):
        img = cv2.imread('./Q2_Image/Cat.png')
        cv2.imshow('original', img)
        imgn = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imshow('Blured', imgn)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnBilateralClicked(self):
        img = cv2.imread('./Q2_Image/Cat.png')
        cv2.imshow('original', img)
        imgn = cv2.bilateralFilter(img, 9, 90, 90)
        cv2.imshow('Blured', imgn)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #part 3
    def btnGaussianFilterClicked(self):
        cv2.imshow('original', self.img3)
        cv2.imshow('grayscale', self.img3_gray)
        cv2.imshow('gaussian blur', self.img3_gau)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnSobelXClicked(self):
        cv2.imshow('Sobel X', self.img_sx)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnSobelYClicked(self):
        cv2.imshow('Sobel Y', self.img_sy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btnMagnitudeClicked(self):
        rows, cols = self.img3_gau.shape
        dst = np.zeros(self.img3_gau.shape)
        for i in range(rows - 2):
            for j in range(cols - 2):
                dst[i+1, j+1] = abs(int(self.img_sx[i+1, j+1])*int(self.img_sx[i+1, j+1]) + int(self.img_sy[i+1, j+1])*int(self.img_sy[i+1, j+1]))**0.5

        dst = dst.astype(np.uint8)
        cv2.imshow('magnitude', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #part 4
    def transform(self):
        img = cv2.imread('./Q4_Image/Parrot.png')
        rows, cols, other = img.shape
        cv2.imshow('original', img)

        try:
            angle = float(self.lineEdit.text())
            scale = float(self.lineEdit_2.text())
            tx = float(self.lineEdit_3.text())
            ty = float(self.lineEdit_4.text())
            dst = img

            mTranslation = np.float32([[1, 0, tx], [0, 1, ty]])
            dst = cv2.warpAffine(dst, mTranslation, (cols, rows))

            mRotation = cv2.getRotationMatrix2D((160+tx, 84+ty), angle, scale)
            dst = cv2.warpAffine(dst, mRotation, (cols, rows))

            cv2.imshow('Image RST', dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except:
            print('Invalid input!!')


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = MainUi()
    ui.show()
    sys.exit(app.exec_())