WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

labels = [
        {'name':'A', 'id':1},
        {'name':'B', 'id':2},
        {'name':'C', 'id':3},
        {'name':'D', 'id':4},
        {'name':'E', 'id':5}]

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 



import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'











import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections,prediction_dict




import cv2 
import numpy as np
import six
import time




import sys
import time
import serial
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog, QMessageBox, QLabel
from Giao_dien_test import Ui_MainWindow
from PyQt5 import QtWidgets

from Sub_page_5_nckh import Ui_Form
import cv2
import speech_recognition

serialcom = serial.Serial('COM3',9600)
serialcom.timeout = 1

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.trangthai = False
        self.uic.label_10.setAlignment(QtCore.Qt.AlignCenter)
        # khai bao nut nhan
        self.uic.pushButton.clicked.connect(self.page_chon_che_do)
        self.uic.pushButton_3.clicked.connect(self.page_hoc_chu)
        self.uic.pushButton_4.clicked.connect(self.page_kiem_tra)
        self.uic.pushButton_5.clicked.connect(self.page_phien_dich)
        self.uic.pushButton_13.clicked.connect(self.page_chon_che_do)
        self.uic.pushButton_54.clicked.connect(self.page_chon_che_do)

        self.uic.pushButton_32.clicked.connect(self.dung_video_page_chon_che_do)

        self.uic.pushButton_52.clicked.connect(self.open_sub_page_5)
        self.uic.pushButton_53.clicked.connect(self.close_sub_page_5)

        #Khai báo chữ cái
        self.uic.pushButton_2.clicked.connect(self.Gui_chu_A)
        self.uic.pushButton_33.clicked.connect(self.Gui_chu_AW)
        self.uic.pushButton_41.clicked.connect(self.Gui_chu_AA)
        self.uic.pushButton_7.clicked.connect(self.Gui_chu_B)
        self.uic.pushButton_8.clicked.connect(self.Gui_chu_C)
        self.uic.pushButton_6.clicked.connect(self.Gui_chu_D)
        self.uic.pushButton_42.clicked.connect(self.Gui_chu_DD)
        self.uic.pushButton_9.clicked.connect(self.Gui_chu_E)
        self.uic.pushButton_43.clicked.connect(self.Gui_chu_EE)
        self.uic.pushButton_10.clicked.connect(self.Gui_chu_G)
        self.uic.pushButton_11.clicked.connect(self.Gui_chu_H)
        self.uic.pushButton_12.clicked.connect(self.Gui_chu_I)
        self.uic.pushButton_15.clicked.connect(self.Gui_chu_K)
        self.uic.pushButton_19.clicked.connect(self.Gui_chu_L)
        self.uic.pushButton_17.clicked.connect(self.Gui_chu_M)
        self.uic.pushButton_20.clicked.connect(self.Gui_chu_N)
        self.uic.pushButton_18.clicked.connect(self.Gui_chu_O)
        self.uic.pushButton_44.clicked.connect(self.Gui_chu_OO)
        self.uic.pushButton_45.clicked.connect(self.Gui_chu_OW)
        self.uic.pushButton_21.clicked.connect(self.Gui_chu_P)
        self.uic.pushButton_16.clicked.connect(self.Gui_chu_Q)
        self.uic.pushButton_14.clicked.connect(self.Gui_chu_R)
        self.uic.pushButton_38.clicked.connect(self.Gui_chu_S)
        self.uic.pushButton_22.clicked.connect(self.Gui_chu_T)
        self.uic.pushButton_31.clicked.connect(self.Gui_chu_U)
        self.uic.pushButton_46.clicked.connect(self.Gui_chu_UW)
        self.uic.pushButton_28.clicked.connect(self.Gui_chu_V)
        self.uic.pushButton_34.clicked.connect(self.Gui_chu_X)
        self.uic.pushButton_27.clicked.connect(self.Gui_chu_Y)
        self.uic.pushButton_47.clicked.connect(self.Gui_chu_sac)
        self.uic.pushButton_48.clicked.connect(self.Gui_chu_huyen)
        self.uic.pushButton_49.clicked.connect(self.Gui_chu_hoi)
        self.uic.pushButton_50.clicked.connect(self.Gui_chu_nga)
        self.uic.pushButton_51.clicked.connect(self.Gui_chu_nang)

        #khai bao cac chu so
        self.uic.pushButton_36.clicked.connect(self.Gui_so_0)
        self.uic.pushButton_25.clicked.connect(self.Gui_so_1)
        self.uic.pushButton_29.clicked.connect(self.Gui_so_2)
        self.uic.pushButton_30.clicked.connect(self.Gui_so_3)
        self.uic.pushButton_23.clicked.connect(self.Gui_so_4)
        self.uic.pushButton_26.clicked.connect(self.Gui_so_5)
        self.uic.pushButton_35.clicked.connect(self.Gui_so_6)
        self.uic.pushButton_37.clicked.connect(self.Gui_so_7)
        self.uic.pushButton_40.clicked.connect(self.Gui_so_8)
        self.uic.pushButton_39.clicked.connect(self.Gui_so_9)


        #page 4
        self.uic.pushButton_24.clicked.connect(self.Cau_ke_tiep)
        self.thread = {}


    
        


    def page_chon_che_do(self):
        self.uic.stackedWidget.setCurrentWidget(self.uic.page_2)
    def page_hoc_chu(self):
        self.uic.stackedWidget.setCurrentWidget(self.uic.page_3)
    def page_kiem_tra(self):
        self.uic.stackedWidget.setCurrentWidget(self.uic.page_4)
    def page_phien_dich(self):
        self.uic.stackedWidget.setCurrentWidget(self.uic.page_5)
    def open_sub_page_5(self):

        self.Second_window = QtWidgets.QMainWindow()
        self.uic1 = Ui_Form()
        self.uic1.setupUi(self.Second_window)
        self.Second_window.show()

        self.thread[2] = speech_to_text(index=2)
        self.thread[2].start()
        self.thread[2].signal_2.connect(self.show_text_from_speech)



    def show_text_from_speech(self, text):
        TEXT = text
        if TEXT != "":
            print(TEXT)
            self.uic.label_10.setText(TEXT)
            self.Second_window.close()



    def close_sub_page_5(self):
        pass

    #Gửi các chữ cái
    def Gui_chu_A(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/a_2.png'))

        gia_tri_serial = "a"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())

    def Gui_chu_AW(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/ă.png'))

        gia_tri_serial = "aw"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_AA(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/â_2.png'))
        gia_tri_serial = "aa"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_B(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/b.png'))
        gia_tri_serial = "b"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())

    def Gui_chu_C(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/c.png'))
        gia_tri_serial = "c"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_D(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/d.png'))
        gia_tri_serial = "d"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_DD(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/đ.png'))
        gia_tri_serial = "dd"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_E(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/e.png'))
        gia_tri_serial = "e"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_EE(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/ê.png'))
        gia_tri_serial = "ee"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_G(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/g.png'))
        gia_tri_serial = "g"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_H(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/h.png'))
        gia_tri_serial = "h"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_I(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/i.png'))
        gia_tri_serial = "i"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_K(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/k.png'))
        gia_tri_serial = "k"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_L(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/l.png'))
        gia_tri_serial = "l"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_M(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/m.png'))
        gia_tri_serial = "m"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_N(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/n.png'))
        gia_tri_serial = "n"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_O(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/o.png'))
        gia_tri_serial = "o"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_OW(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/ơ.png'))
        gia_tri_serial = "ow"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_OO(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/ô.png'))
        gia_tri_serial = "oo"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_P(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/p.png'))
        gia_tri_serial = "p"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_Q(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/q.png'))
        gia_tri_serial = "q"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_R(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/r.png'))
        gia_tri_serial = "r"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_S(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/s.png'))
        gia_tri_serial = "s"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_T(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/t.png'))
        gia_tri_serial = "t"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_U(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/u.png'))
        gia_tri_serial = "u"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_UW(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/ư.png'))
        gia_tri_serial = "uw"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_V(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/v.png'))
        gia_tri_serial = "v"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_X(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/x.png'))
        gia_tri_serial = "x"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_Y(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/y.png'))
        gia_tri_serial = "y"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_sac(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/dau sac.png'))
        gia_tri_serial = "sac"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_huyen(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/dau huyen.png'))
        gia_tri_serial = "huyen"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_hoi(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/dau hoi.png'))
        gia_tri_serial = "hoi"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_nga(self):
        self.uic.label_4.setPixmap(QPixmap('Chu_cai_va_so/dau ngã.png'))
        gia_tri_serial = "nga"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_chu_nang(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/dau nang.png"))
        gia_tri_serial = "nang"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())

    #Gửi các chữ số
    def Gui_so_0(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/0.png"))
        gia_tri_serial = "0"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_1(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/1.png"))
        gia_tri_serial = "1"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_2(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/2.png"))
        gia_tri_serial = "2"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_3(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/3.png"))
        gia_tri_serial = "3"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_4(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/4.png"))
        gia_tri_serial = "4"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_5(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/5.png"))
        gia_tri_serial = "5"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_6(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/6.png"))
        gia_tri_serial = "6"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_7(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/7.png"))
        gia_tri_serial = "7"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_8(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/8.png"))
        gia_tri_serial = "8"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())
    def Gui_so_9(self):
        self.uic.label_4.setPixmap(QPixmap("Chu_cai_va_so/9.png"))
        gia_tri_serial = "9"
        print(gia_tri_serial)
        serialcom.write(gia_tri_serial.encode())


    #page 4
    def stop_capture_video(self):
        self.thread[1].stop()
        
    def dung_video_page_chon_che_do(self):
        self.uic.stackedWidget.setCurrentWidget(self.uic.page_2)
        if self.trangthai == True:
            self.stop_capture_video()
            self.trangthai = False
            
    def closeEvent(self, event):
        if self.trangthai == True:
            self.stop_capture_video()
            self.trangthai = False

        reply = QMessageBox.question(self, 'Thoát chương trình', 'Bạn có chắc chắn muốn thoát chương trình không?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
            print('Thoát chương trình')
        else:
            event.ignore()
    def Cau_ke_tiep(self):

        self.thread[1] = capture_video(index=1)
        self.thread[1].start()
        self.thread[1].signal.connect(self.show_wedcam)
        self.thread[1].signal_1.connect(self.chuyen_cau_hoi)
        self.trangthai = True


    def show_wedcam(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.label_7.setPixmap(qt_img)
    def chuyen_cau_hoi(self, number_question):
        if number_question == 1:
            self.uic.label_6.setPixmap(QPixmap("Chu_cai_va_so/b.png"))
            
        if number_question == 2:
            self.cau_tra_loi_dung()
            self.uic.label_12.setPixmap(QPixmap("Chu_cai_va_so/Nut_T4_2.png"))
            self.uic.label_6.setPixmap(QPixmap("Chu_cai_va_so/c.png"))
        if number_question == 3:
            self.cau_tra_loi_dung()
            self.uic.label_13.setPixmap(QPixmap("Chu_cai_va_so/Nut_T4_2.png"))
            self.uic.label_6.setPixmap(QPixmap("Chu_cai_va_so/a_2.png"))
        if number_question == 4:
            self.cau_tra_loi_dung()
            self.uic.label_14.setPixmap(QPixmap("Chu_cai_va_so/Nut_T4_2.png"))
            self.uic.label_6.setPixmap(QPixmap("Chu_cai_va_so/e.png"))
        if number_question == 5:
            self.cau_tra_loi_dung()
            self.uic.label_14.setPixmap(QPixmap("Chu_cai_va_so/Nut_T4_2.png"))
            self.uic.label_6.setPixmap(QPixmap("Chu_cai_va_so/d.png"))

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(235, 178, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def cau_tra_loi_dung(self):
        self.thread[3] = show_video_tra_loi_dung(index=3)
        self.thread[3].start()
        self.thread[3].signal_3.connect(self.show_video)

    def show_video(self, cv_vid):
        """Updates the image_label with a new opencv image"""
        qt_vid = self.convert_video(cv_vid)
        self.uic.label_8.setPixmap(qt_vid)


    def convert_video(self, cv_vid):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_vid, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1282, 462, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class capture_video(QThread):
    signal = pyqtSignal(np.ndarray)
    signal_1 = pyqtSignal(int)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super().__init__()
    
    def run(self):
        category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
        # Setup capture
        
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        TT_cau_hoi_tam_thoi=1
        TT_cau_hoi = 0 
        while True:
            ret, frame = cap.read()
            image_np = np.array(frame)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Things to try:
            # Flip horizontally
            # image_np = np.fliplr(image_np).copy()

            # Convert image to grayscale
            # image_np = np.tile(
            #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections , prediction_dict  = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            def get_classes_name_and_scores(
                boxes,
                classes,
                scores,
                category_index,
                max_boxes_to_draw=20,
                min_score_thresh=.9): # returns bigger than 90% precision
                display_str = {}
                if not max_boxes_to_draw:
                    max_boxes_to_draw = boxes.shape[0]
                for i in range(min(max_boxes_to_draw, boxes.shape[0])):
                    if scores is None or scores[i] > min_score_thresh:
                        if classes[i] in six.viewkeys(category_index):
                            display_str['name'] = category_index[classes[i]]['name']
                            display_str['score'] = '{}%'.format(int(100 * scores[i]))

                return display_str

            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.5,
                        agnostic_mode=False)

            


            result_predict = get_classes_name_and_scores(
              detections['detection_boxes'],
              detections['detection_classes']+label_id_offset,
              detections['detection_scores'],
              category_index)
            if (result_predict != {}):
                result_name = result_predict['name']
                result_score = result_predict['score'].rstrip("%")
                result_score = int(result_score)
            else:
                result_name=""
                result_score = 0

            if (result_name == "B") & (result_score > 95) & (TT_cau_hoi == 1)  :
                time.sleep(0.02)
                if (result_name == "B") & (result_score > 95) & (TT_cau_hoi == 1):
                    time.sleep(0.02)
                    if (result_name == "B") & (result_score > 95) & (TT_cau_hoi == 1):
                        time.sleep(0.02)
                        if (result_name == "B") & (result_score > 95) & (TT_cau_hoi == 1):
                            time.sleep(0.02)
                            if (result_name == "B") & (result_score > 95) & (TT_cau_hoi == 1):
                                TT_cau_hoi_tam_thoi = 2
                                print("ban da tra loi dung chữ B")
            if (result_name == "C") & (result_score > 95) & (TT_cau_hoi == 2)  :
                time.sleep(0.02)
                if (result_name == "C") & (result_score > 95) & (TT_cau_hoi == 2):
                    time.sleep(0.02)
                    if (result_name == "C") & (result_score > 95) & (TT_cau_hoi == 2):
                        time.sleep(0.02)
                        if (result_name == "C") & (result_score > 95) & (TT_cau_hoi == 2):
                            time.sleep(0.02)
                            if (result_name == "C") & (result_score > 95) & (TT_cau_hoi == 2):
                                TT_cau_hoi_tam_thoi = 3
                                print("ban da tra loi dung chữ C")
            if (result_name == "A") & (result_score > 95) & (TT_cau_hoi == 3)  :
                time.sleep(0.02)
                if (result_name == "A") & (result_score > 95) & (TT_cau_hoi == 3):
                    time.sleep(0.02)
                    if (result_name == "A") & (result_score > 95) & (TT_cau_hoi == 3):
                        time.sleep(0.02)
                        if (result_name == "A") & (result_score > 95) & (TT_cau_hoi == 3):
                            time.sleep(0.02)
                            if (result_name == "A") & (result_score > 95) & (TT_cau_hoi == 3):
                                TT_cau_hoi_tam_thoi = 4
                                print("ban da tra loi dung chữ A")
            if (result_name == "E") & (result_score > 90) & (TT_cau_hoi == 4)  :
                time.sleep(0.02)
                if (result_name == "E") & (result_score > 90) & (TT_cau_hoi == 4):
                    time.sleep(0.02)
                    if (result_name == "E") & (result_score > 90) & (TT_cau_hoi == 4):
                        time.sleep(0.02)
                        if (result_name == "E") & (result_score > 90) & (TT_cau_hoi == 4):
                            time.sleep(0.02)
                            if (result_name == "E") & (result_score > 90) & (TT_cau_hoi == 4):
                                TT_cau_hoi_tam_thoi = 5
                                print("ban da tra loi dung chữ E")
            if (result_name == "D") & (result_score > 95) & (TT_cau_hoi == 5)  :
                time.sleep(0.02)
                if (result_name == "D") & (result_score > 95) & (TT_cau_hoi == 5):
                    time.sleep(0.02)
                    if (result_name == "D") & (result_score > 95) & (TT_cau_hoi == 5):
                        time.sleep(0.02)
                        if (result_name == "D") & (result_score > 95) & (TT_cau_hoi ==5):
                            time.sleep(0.02)
                            if (result_name == "D") & (result_score > 95) & (TT_cau_hoi == 5):
                                TT_cau_hoi_tam_thoi = 6
                                print("ban da tra loi dung chữ D")
            if ret:
                self.signal.emit(image_np_with_detections)
            if TT_cau_hoi_tam_thoi != TT_cau_hoi :
                TT_cau_hoi = TT_cau_hoi_tam_thoi
                self.signal_1.emit(TT_cau_hoi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                break
    def stop(self):
        print("stop threading", self.index)
        self.terminate()

class speech_to_text(QThread):
    signal_2 = pyqtSignal(str)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super().__init__()

    def run(self):
        robot_ear = speech_recognition.Recognizer()
        with speech_recognition.Microphone() as mic:
            print("Robot: Tôi đang lắng nghe bạn")
            audio = robot_ear.record(mic, duration=4)
        try:
            you = robot_ear.recognize_google(audio, language="vi")
        except:
            you = ""

        self.signal_2.emit(you)

    def stop(self):
        print("stop threading", self.index)
        self.terminate()

class show_video_tra_loi_dung(QThread):
    signal_3 = pyqtSignal(np.ndarray)
    def __init__(self, index):
        self.index = index
        print("start threading", self.index)
        super().__init__()

    def run(self):
        cap = cv2.VideoCapture("Chu_cai_va_so/CHINH_XAC_4.mp4")  # 'D:/8.Record video/My Video.mp4'
        while True:
            ret, cv_vid = cap.read()
            if ret:
                self.signal_3.emit(cv_vid)
    def stop(self):
        print("stop threading", self.index)
        self.terminate()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())