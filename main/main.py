import tkinter as tk
import PIL.Image
import PIL.ImageTk

from .videoCapture import *

class Main:
    """Class utama yang berfungsi untuk menampilkan GUI"""
    def __init__(self, title, video_source=0):
        self.__gui = tk.Tk()

        self.vid = FaceRecognition(video_source, 100)
        self.__config(title, ("%dx%d" % (self.vid.width, self.vid.height + 50)))
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
        
        self.__button();
        
    def __button(self):
        frame = tk.Frame(self.leftFrame)
        frame.pack(fill=tk.X, side=tk.TOP,
                   pady=(0, 3), ipadx=2)

        button_connect = tk.Button(
            frame, text=" Tambah dataset ", command=self.__add, width=15)
        
        button_connect.pack(side=tk.LEFT, padx=10, pady=2)
        
    def __webcamView(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.__photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.videoFrame.configure(image=self.__photo)

        self.__gui.after(1, self.__webcamView)
        
    def __add(self):
        return