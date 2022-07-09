from tkinter import ttk
import tkinter as tk
import datetime
import PIL.Image
import PIL.ImageTk

from .videoCapture import *

class Main:
    """Class utama yang berfungsi untuk menampilkan GUI"""
    __lastDataTotal = 0

    __listData = []
    
    def __init__(self, title, defSize, video_source=0):
        self.__gui = tk.Tk()

        self.vid = FaceRecognition(video_source)
        self.__config(title, defSize)
        self.__webcamView()
        self.__logView()
        self.addLog('20.11.1234', 'John Doe')

    # def setFont(self, size, font='TkDefaultFont'):
        # def_font = tk.font.nametofont(font)
        # def_font.config(size=size)
    
    def start(self):
        self.__gui.mainloop()

    def __config(self, title, defSize):
        self.__gui.title(title)
        self.__gui.geometry(defSize)

        self.__gui.columnconfigure(0, weight=1)
        self.__gui.rowconfigure(0, weight=1)

        self.leftFrame = tk.Frame(self.__gui)
        self.leftFrame.pack(fill=tk.Y, side=tk.LEFT,
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
            frame, text=" Mulai ", command=self.connect, width=10)
        button_disconenct = tk.Button(
            frame, text="Berhenti", command=self.disconnect, width=10)
        
        button_connect.pack(side=tk.LEFT, padx=2, pady=2)
        button_disconenct.pack(side=tk.RIGHT, padx=2, pady=2)
        
    def __logView(self):
        logFrame = tk.Frame(self.__gui)
        logFrame.pack(fill=tk.Y, side=tk.RIGHT,
                      pady=(0, 3), ipadx=300, expand=1)

        scrollbar = tk.Scrollbar(logFrame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.__logList = ttk.Treeview(logFrame)
        self.__logList.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        self.__logList.config(yscrollcommand=scrollbar.set)

        scrollbar.config(command=self.__logList.yview)

        self.__logList['columns'] = ('Number', 'Time', 'NIM', 'Name')

        self.__logList.column('#0', width=0, stretch=tk.NO)
        self.__logList.column('Number', stretch=tk.NO,
                              anchor=tk.CENTER, width=50)
        self.__logList.column('Time', stretch=tk.NO,
                              anchor=tk.CENTER, width=120)
        self.__logList.column('NIM', stretch=tk.NO,
                              anchor=tk.CENTER, width=120)
        self.__logList.column('Name', width=200)

        self.__logList.heading('#0', text='', anchor=tk.CENTER)
        self.__logList.heading('Number', text='No', anchor=tk.CENTER)
        self.__logList.heading('Time', text='Time', anchor=tk.CENTER)
        self.__logList.heading('NIM', text='NIM', anchor=tk.CENTER)
        self.__logList.heading('Name', text='Name', anchor=tk.CENTER)

    def __webcamView(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.__photo = PIL.ImageTk.PhotoImage(
                image=PIL.Image.fromarray(frame))
            self.videoFrame.configure(image=self.__photo)

        self.__gui.after(5, self.__webcamView)

    def addLog(self, nim, name):
        timeNow = datetime.datetime.now()
        timestamp = timeNow.strftime("%H:%M:%S")
        
        self.__lastDataTotal += 1
        
        self.__listData.append({"nim" : nim, "name" : name, "time" : timestamp})
        
        self.__logList.insert('', 'end', text='',
                              values=(str(self.__lastDataTotal), self.__listData[-1]["time"], self.__listData[-1]["nim"], self.__listData[-1]["name"]))

    def connect(self):
        return
    
    def disconnect(self):
        return
