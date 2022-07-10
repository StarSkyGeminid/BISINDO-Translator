import tkinter as tk
from tkinter import ttk
import PIL.Image
import PIL.ImageTk
import threading as th
from .videoCapture import *
from .datasetMaker import *
import time

class AddDataset:
    """Class yang berfungsi untuk menampilkan GUI untuk menambah data set"""

    def __init__(self, title, video_source=0):
        self.__gui = tk.Tk()

        self.value = 'A'
        self.write = False
        
        self.vid = FaceRecognition(video_source, 100)
        self.__config(title, ("%dx%d" %
                      (self.vid.width, self.vid.height + 50)))
        self.__webcamView()

    def start(self):
        self.__gui.mainloop()

    def __config(self, title, defSize):
        self.__gui.title(title)
        self.__gui.geometry(defSize)

        self.__gui.columnconfigure(0, weight=1)
        self.__gui.rowconfigure(0, weight=1)

        self.leftFrame = tk.Frame(self.__gui)
        self.leftFrame.pack(fill=tk.BOTH, side=tk.LEFT,
                            pady=(0, 3), ipadx=2)

        self.videoFrame = tk.Label(self.leftFrame, width=int(
            self.vid.width), height=int(self.vid.height))
        self.videoFrame.pack(side=tk.TOP, padx=2, pady=2)

        controllerFrame = tk.Frame(self.leftFrame)
        controllerFrame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 3), ipadx=2)

        self.__button(controllerFrame)
        self.__inputType(controllerFrame)
        self.displayMsg(controllerFrame)
        
    def __button(self, frame):
        button_connect = tk.Button(
            frame, text=" Mulai Menambah Dataset ", command=self.__add, width=35)

        button_connect.pack(side=tk.LEFT, padx=10, pady=2)
        
    def __inputType(self, frame):
        alfabertLabel = tk.Label(frame, text="Huruf  :")
        self.alfabertBox = ttk.Combobox(frame, values=self.value, width=10)
        
        self.alfabertBox['values'] = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
        self.alfabertBox.current(0)
        
        self.alfabertBox.pack(side=tk.RIGHT, padx=2, pady=2)
        alfabertLabel.pack(side=tk.RIGHT, padx=2, pady=2)

    def displayMsg(self, frame):
        self.timoutLabel = tk.Label(frame, text='')
        self.timoutLabel.pack(side=tk.LEFT, padx=2, pady=2)
    
    def countDown(self):
        while True:
            self.count = self.count - 1
            
            if self.count < -5:
                self.timoutLabel['text'] = (
                    "Sampel berhasil ditambahkan!")
                self.write = False
                maker = DatasetMaker()
                maker.train()
                
                break
            elif self.count >= 0:
                self.timoutLabel['text'] = ("Mengambil sampel dalam " + str(self.count) +" detik")
            else:
                self.timoutLabel['text'] = (
                    "Sedang mengambil sampel...")
                self.write = True

            time.sleep(1)
        
        
    def __webcamView(self):
        ret, frame = self.vid.get_frame(save=self.write, text=self.value)

        if ret:
            self.__photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.videoFrame.configure(image=self.__photo)

        self.__gui.after(1, self.__webcamView)

    def __add(self):
        self.count = 5

        self.countThread = th.Thread(target=self.countDown)
        self.countThread.start()
        return
