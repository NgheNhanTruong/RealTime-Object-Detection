import sys

import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog,QApplication
from PyQt5.uic import loadUi
import datetime

class tehseeencode(QDialog):
    def __init__(self):
        super(tehseeencode, self).__init__()
        loadUi('student.ui',self)
        self.logic=0
        self.value=1
        self.SHOW.clicked.connect(self.onClicked)

        self.TEXT.setText("bam SHOW de bat webcam")

    @pyqtSlot()
    def onClicked(self):
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):
            ret,frame=cap.read()
            if ret == True:
                print("here")
                self.displayImage(frame,1)
                cv2.waitKey()


        cap.release()
        cv2.destroyAllWindows()



    def displayImage(self,img,window=1):
        qformat = QImage.Format_Indexed8

        if len(img.shape) ==3:
            if (img.shape[2])==4:
                qformat=QImage.Format_RGBA8888

            else:
                qformat=QImage.Format_RGB888
        img = QImage(img,img.shape[1],img.shape[0],qformat)
        img =img.rgbSwapped()
        self.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

app=QApplication(sys.argv)
window = tehseeencode()
window.show()
try:
    sys.exit(app.exec_())
except:
    print('exit')


