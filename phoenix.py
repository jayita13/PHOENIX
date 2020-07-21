from tkinter import *
from tkinter.ttk import Combobox
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
import time
import tkinter.ttk as ttk
from tkinter import filedialog
from app import display
from main import mainc
from threading import Thread
from video_recorder import start
import cv2


global version

version='1.0'



# THE STARING OF PYTHON CODE
def mainx():
    regwindowx = tk.Tk()
    screen_widthx = regwindowx.winfo_screenwidth()
    # screen_heightx = regwindowx.winfo_screenheight()
    regwindowx.destroy()

    def loading():
        rootx = tk.Tk()
        rootx.iconbitmap(default='IMAGES/home.ico')
        # The image must be stored to Tk or it will be garbage collected.
        rootx.image = tk.PhotoImage(file='IMAGES/front.gif')
        labelx = tk.Label(rootx, image=rootx.image, bg='white')
        rootx.overrideredirect(True)
        rootx.geometry("+450+140")
        # root.lift()
        rootx.wm_attributes("-topmost", True)
        rootx.wm_attributes("-disabled", True)
        rootx.wm_attributes("-transparentcolor", "white")
        labelx.pack()
        labelx.after(500, lambda: labelx.destroy())
        rootx.after(500, lambda: rootx.destroy())  # Destroy the widget after 0.5 seconds
        labelx.mainloop()


    for i in range(0,3):
        loading()







    class Store_DATA_IN_INI():

        # OPTION SELECT POP UP CREATION

        def __init__(self, win):


            load = cv2.imread('IMAGES/covid.jpg', 1)
            cv2imagex1 = cv2.cvtColor(load, cv2.COLOR_BGR2RGBA)
            load = Image.fromarray(cv2imagex1)
            load = load.resize((int(800), int(450)), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = tk.Label(image=render)
            img.image = render
            img.place(x=-1, y=0)            



            def user_video():
                window.destroy()
                display()


            self.b3 = ttk.Button(win, text='CLICK TO START', width=20, command=self.store_INI)
            self.b3.place(x=250, y=300, width=200, height=50)

            button_over_ride = Button(win, height=1, width=1, bg='white', bd=0, command=user_video)
            button_over_ride.place(x=0, y=1)



        def store_INI(self):

            window.destroy()

            mainc()


    window = Tk()
    window.iconbitmap(default='IMAGES/home.ico')
    option_window = Store_DATA_IN_INI(window)
    window.config(background='white')
    window.attributes('-alpha', 0.9)
    window.title('PHOENIX-SOCIAL DISTANCING AND MASK DETECTOR' + version)
    window.geometry("750x450")
    window.mainloop()






if __name__ == '__main__':
    Thread(target=mainx).start()
    Thread(target=start).start()


   