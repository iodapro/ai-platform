import wx
import nstmulti as nst
import threading
import scipy
import os
from multiprocessing import Lock
from multiprocessing import Value
import time
import sys
import styleTransfer as st
import utils
import nstModels

# class responsible for reacting on event caused by button pressing, text typed etc.
class GUIController:
    def __init__(self,frame,interV,deltaV):
        self.nst = st.NeuralStyleTransfer(nstModels.VGG19ForNST())
        self.separator = utils.getSeparator()
        self._contentImagePath = ''
        self.outfilename = ''
        self.styleName = ''
        self.oldValue = 0
        self._styleImagesPaths = []
        self._isNSTButtonLocked = False
        self.value = Value('i',0)
        self.frame = frame
        self.nUpdates = 1000
        self.gauge = wx.TextCtrl(self.frame, size = (620, 50), pos = (10,8*interV+11*deltaV),style=wx.TE_MULTILINE | wx.TE_READONLY)
        if( not os.path.exists('out') ):
            os.mkdir('out')
        if( not os.path.exists('styles') ):
            os.mkdir('styles')
    def onButtonContentImage(self,event):
        frame = wx.Frame(None, -1, '')
        openFileDialog = wx.FileDialog(frame, "Open", "", "", 
                                      "Python files (*.py)|*.jpg", 
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        openFileDialog.ShowModal()
        self._contentImagePath = openFileDialog.GetPath()
        openFileDialog.Destroy()
    def onButtonStyleImages(self,event):
        frame = wx.Frame(None, -1, '')
        openFileDialog = wx.FileDialog(frame, "Open", "", "", 
                                      "Python files (*.py)|*.jpg", 
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        openFileDialog.ShowModal()
        self._styleImagesPaths.append(openFileDialog.GetPath())
        openFileDialog.Destroy()
    # pressing this button runs new thread in daemon regime allowing to perform nst calculation and avalaibility of gui at the same time
    def onButtonNTS(self, event):
        if self._isNSTButtonLocked == True:
            return
        else:
            if self._contentImagePath == '' or (len(self._styleImagesPaths) == 0) or self.outfilename == '':
                return
            self._isNSTButtonLocked = True
            self.nst.isTerminate = False
            outFile = self.outfilename
            self.gauge.SetValue("Output image: " + os.getcwd()+self.separator+"out"+self.separator+outFile+".jpg\n")
            self._p = threading.Thread(target=self.nst.run_style_transfer, args = (self._contentImagePath,self._styleImagesPaths, outFile, self.value, self.nUpdates))
            self._p.daemon = True
            self._p.start()
    # once this button pressed program waiting while isTerminate == True is executed and terminates created thread
    def onButtonStop(self, event):
        if self._isNSTButtonLocked == True:
            self.nst.isTerminate = True
            self._p.join()
            print("Terminated")
            self._isNSTButtonLocked = False
    def OnKeyTyped(self, event): 
      self.outfilename = event.GetString()
    def onstyleTyped(self, event):
        self.styleName = event.GetString()
    def onSaveStyle(self,event):
        f= open('styles'+self.separator+self.styleName+'.style',"w+")
        for el in self._styleImagesPaths:
            f.write("%s\n" % el)
        f.close()
        self.gauge.AppendText("Saved style: " + os.getcwd()+self.separator+"styles"+self.separator+self.styleName+".style")
    def onChooseSavedStyle(self, event):
        frame = wx.Frame(None, -1, '')
        openFileDialog = wx.FileDialog(frame, "Open", "", "", 
                                      "Style files (*.style)|*.style", 
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        openFileDialog.ShowModal()
        pathToStyleFile = openFileDialog.GetPath()
        f = open(pathToStyleFile, 'r') 
        rawStylesString = f.readlines()
        for el in range(len(rawStylesString)):
            rawStylesString[el] = rawStylesString[el].rstrip()
        self._styleImagesPaths = rawStylesString
        openFileDialog.Destroy()
    def onUpdates(self,event):
        try:
            self.nUpdates = int(event.GetString())
        except:
            self.nUpdates = 1000
    # before exiting the program check is performed whther additional thread is running, and if so it terminates it. "isTerminate" is considered to be shared variable for both threads.
    def onExit(self, event):
        if self._isNSTButtonLocked == True:
            self.nst.isTerminate = True
            self._p.join()
        self.frame.Destroy()
        sys.exit()

# class responsible for visualization GUI. There are a lot of magic numbers defined, which control where and of which size the graphical item is put.
class GUIBuilder:
    def __init__(self):
        self.width = 160
        self.height = 30
        self.deltaV = 25
        self.interV = 10

        self.stepSize = (250,self.height)
        self.whSize = (self.width, self.height)
        self.leftMargin = 10
        self.middleMargin = 240
        self.rightMargin = 470

        self.app = wx.App()
        self.frame = wx.Frame(None, -1, 'NST',style=wx.DEFAULT_FRAME_STYLE & (~wx.CLOSE_BOX) & (~wx.MAXIMIZE_BOX))
        self.frame.SetDimensions(0,0,680,500)
    def initializeArchitecture(self):
        self.initStaticText()
        self.initButtons()
        self.initTextCtrl()
    def initStaticText(self):
        wx.StaticText(self.frame, -1, "Step1: choose content image", size=self.stepSize, pos=(self.middleMargin,self.interV))
        wx.StaticText(self.frame, -1, "Step2: choosing style", size=self.stepSize, pos=(self.middleMargin,2*self.interV+3*self.deltaV))
        wx.StaticText(self.frame, -1, "Step3: choosing output file name", size=self.stepSize, pos=(self.middleMargin,4*self.interV+6*self.deltaV))
        wx.StaticText(self.frame, -1, "Output file name", size=self.whSize,pos=(self.middleMargin,4*self.interV+7*self.deltaV)) 
        wx.StaticText(self.frame, -1, "Step4: NST controller", size=self.stepSize, pos=(self.middleMargin,7*self.interV+9*self.deltaV))
        wx.StaticText(self.frame, -1, "Number of updates", size=self.whSize,pos=(self.rightMargin,self.interV+self.deltaV)) 
        wx.StaticText(self.frame, -1, "New style name", size=self.whSize,pos=(self.rightMargin,3*self.interV+3*self.deltaV))  
    def initButtons(self):
        self.buttonContent = wx.Button(self.frame, label = 'Choose content image', size=self.whSize, pos=(self.leftMargin,self.interV+2*self.deltaV))
        self.buttonStyle = wx.Button(self.frame, label = 'Choose style images', size=self.whSize, pos = (self.leftMargin, 3*self.interV+4*self.deltaV))
        self.buttonNST = wx.Button(self.frame, label = 'Start NST', size=self.whSize, pos = (self.leftMargin,7*self.interV+10*self.deltaV))
        self.buttonChooseStyle = wx.Button(self.frame, label = 'Choose saved style', size=self.whSize, pos=(self.middleMargin,3*self.interV+4*self.deltaV))
        self.buttonSave = wx.Button(self.frame, label = 'Save style', size=self.whSize, pos = (self.rightMargin,3*self.interV+5*self.deltaV))
        self.buttonStop = wx.Button(self.frame, label = 'Stop NST', size=self.whSize, pos = (self.rightMargin,7*self.interV+10*self.deltaV))
        self.buttonExit = wx.Button(self.frame, label = 'Exit', size = self.whSize, pos = (self.rightMargin,410))
    def initTextCtrl(self):
        self.t1 = wx.TextCtrl(self.frame,size=self.whSize,pos=(self.middleMargin,4*self.interV+8*self.deltaV))
        self.tI = wx.TextCtrl(self.frame,size=self.whSize,pos=(self.rightMargin,self.interV+2*self.deltaV))
        self.tI.SetValue(str(1000))
        self.t2 = wx.TextCtrl(self.frame,size=self.whSize,pos=(self.rightMargin,3*self.interV+4*self.deltaV))
    def initGUIController(self):
        self.gui = GUIController(self.frame,self.interV,self.deltaV)
    def bindArchitectureAndEvents(self):
        self.initGUIController()
        self.initializeArchitecture()
        self.buttonContent.Bind(wx.EVT_BUTTON, self.gui.onButtonContentImage)
        self.buttonStyle.Bind(wx.EVT_BUTTON, self.gui.onButtonStyleImages)
        self.buttonNST.Bind(wx.EVT_BUTTON, self.gui.onButtonNTS)
        self.buttonStop.Bind(wx.EVT_BUTTON, self.gui.onButtonStop)
        self.buttonSave.Bind(wx.EVT_BUTTON, self.gui.onSaveStyle)
        self.buttonChooseStyle.Bind(wx.EVT_BUTTON, self.gui.onChooseSavedStyle)
        self.buttonExit.Bind(wx.EVT_BUTTON, self.gui.onExit)
        self.t1.Bind(wx.EVT_TEXT, self.gui.OnKeyTyped)
        self.t2.Bind(wx.EVT_TEXT, self.gui.onstyleTyped)
        self.tI.Bind(wx.EVT_TEXT, self.gui.onUpdates)
    def runGUI(self):
        self.bindArchitectureAndEvents()
        self.frame.Show()
        self.frame.Centre()
        self.app.MainLoop()
