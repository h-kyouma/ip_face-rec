import cv2
import numpy as np
from PySide import QtGui, QtCore
import os
import sys

class onImage:       
    def loadImage(self,path):
        img = cv2.imread(path)
        return img
        
    def loadCascades(self,facePath,eyePath):
        face_cascade = cv2.CascadeClassifier(facePath)
        eye_cascade = cv2.CascadeClassifier(eyePath)
        return face_cascade,eye_cascade
           
    def recOnImage(self,img,face_cascade,eye_cascade):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        cv2.imwrite(os.path.expanduser('~/Documents/FT/temp.png'),img)
        
    def run(self):
        img = self.loadImage(os.path.expanduser('~/Documents/FT/temp.png'))
        face_cascade,eye_cascade = self.loadCascades('haarcascade_frontalface_default.xml','haarcascade_eye.xml')
        self.recOnImage(img,face_cascade,eye_cascade)
    
class liveTracking:
    def __init__(self):
        self.track_len = 3
        self.detect_interval = 5
        self.tracks = []
        self.cap = cv2.VideoCapture(0)
        self.frame_idx = 0
        self.run()
        
    def loadCascades(self,facePath,eyePath):
        face_cascade = cv2.CascadeClassifier(facePath)
        eye_cascade = cv2.CascadeClassifier(eyePath)
        return face_cascade,eye_cascade
        
    def loadParams(self):
        lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
                       
        return lk_params,feature_params

    def run(self):
        face_cascade,eye_cascade = self.loadCascades('haarcascade_frontalface_default.xml','haarcascade_eye.xml')
        lk_params,feature_params = self.loadParams()
        show_eyes = False
        
        while True:
            ret, frame = self.cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            for (faceX, faceY, faceW, faceH) in faces:
                cv2.ellipse(vis, ((faceX+faceX+faceW)/2,(faceY+faceY+faceH)/2), ((faceX+faceW)/5,(faceY+faceH)/3),0,0,360,(255,0,0),2)
                if show_eyes:
                    roi_gray = frame_gray[faceY:faceY+faceH, faceX:faceX+faceW]
                    roi_color = vis[faceY:faceY+faceH, faceX:faceX+faceW]
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
                    for (eyesX,eyesY,eyesW,eyesH) in eyes:
                        cv2.rectangle(roi_color, (eyesX,eyesY), (eyesX+eyesW, eyesY+eyesH), (0,255,0), 2)
            cv2.putText(vis,'Toggle 1 for eyes detection',(0,460),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,255),2)
            if len(faces)==0:
                cv2.putText(vis,'No face detected',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    try:
                        if faceX<x<faceX+faceW and faceY<y<faceY+faceH:
                            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    except:
                        continue                   
                self.tracks = new_tracks
                if faceX<x<faceX+faceW and faceY<y<faceY+faceH:
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('Face Tracking', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                cv2.destroyAllWindows()
                break
            if ch == ord('1'):
                show_eyes = not show_eyes
                
class GUI(QtGui.QMainWindow):
    def __init__(self):
        super(GUI,self).__init__()
        
        self.scaleFactor = 0.0
        self.onimage = onImage()
        
        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored,QtGui.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        
        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)
        
        self.createActions()
        self.createMenus()
        
        self.setWindowTitle('Face recognition')
        self.resize(1360,768)
        
    def open(self):  
        if os.path.exists(os.path.expanduser('~/Documents/FT/temp.png')):
            os.remove(os.path.expanduser('~/Documents/FT/temp.png'))
            
        fileName,_ = QtGui.QFileDialog.getOpenFileName(self,'Open File',QtCore.QDir.currentPath())
        if fileName:
            image = QtGui.QImage(fileName)
            if image.isNull():
                QtGui.QMessageBox.information(self,'Program','Cannot load %s.' % fileName)
                return
                
            image.save(os.path.expanduser('~/Documents/FT/temp.png'))
            self.picturetracking()
            imageNew = QtGui.QImage(os.path.expanduser('~/Documents/FT/temp.png'))
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(imageNew))
            self.createfolder()
            
            self.scaleFactor = 1.0
            
            self.fitToWindowAct.setEnabled(True)
            self.updateActions()
            
            if not self.fitToWindowAct.isChecked():
                self.imageLabel.adjustSize()
                
    def createfolder(self):
        path = os.path.expanduser('~/Documents/FT/')
        if not os.path.exists(path):
            os.makedirs(path)
        
    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()
            
        self.updateActions()
        
    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0
        
    def zoomIn(self):
        self.scaleImage(1.25)
        
    def zoomOut(self):
        self.scaleImage(0.8)
        
    def picturetracking(self):
        self.onimage.run()
        
    def liveCapture(self):
        self.livetracking = liveTracking()
        
    def createActions(self):
        self.openAct = QtGui.QAction('&Open image...',self,shortcut='Ctrl+O',triggered=self.open)
        
        self.exitAct = QtGui.QAction('&Exit...',self,shortcut='Ctrl+Q',triggered=self.close)
        
        self.fitToWindowAct = QtGui.QAction('&Fit To Window',self,enabled=False,checkable=True,shortcut='Ctrl+F',triggered=self.fitToWindow)
        
        self.normalSizeAct = QtGui.QAction('&Normal Size',self,shortcut='Ctrl+N',enabled=False,triggered=self.normalSize)
        
        self.captureAct = QtGui.QAction('&Capture',self,shortcut='Ctrl+C',triggered=self.liveCapture)
        
        self.zoomInAct = QtGui.QAction('Zoom &In (25%)',self,shortcut='Ctrl++',enabled=False,triggered=self.zoomIn)
        
        self.zoomOutAct = QtGui.QAction('Zoom &Out (25%)',self,shortcut='Ctrl+-',enabled=False,triggered=self.zoomOut)
        
    def createMenus(self):
        self.fileMenu = QtGui.QMenu('&File',self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        
        self.viewMenu = QtGui.QMenu('&View',self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addAction(self.fitToWindowAct)
        
        self.captureMenu = QtGui.QMenu('&Live webcam tracking',self)
        self.captureMenu.addAction(self.captureAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.captureMenu)
        
    def updateActions(self):
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        
    def scaleImage(self,factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)
        
    def adjustScrollBar(self,scrollBar,factor):
        scrollBar.setValue(int(factor*scrollBar.value()+((factor-1)*scrollBar.pageStep()/2)))
                
def main():
    app = QtGui.QApplication.instance()
    
    gui = GUI()
    gui.show()
    
    sys.exit(app.exec_())
    
main()