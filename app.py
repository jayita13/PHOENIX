from tkinter import *
from tkinter.ttk import *
from tkinter.ttk import Combobox
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
from threading import Thread
import tkinter.ttk as ttk
from tkinter import filedialog
import cv2


global version

version='1.0'


def display():

    class INIX():


        def __init__(self, win):

            screen_widthx = win.winfo_screenwidth()

            screen_heightx = win.winfo_screenheight()


            load = cv2.imread('IMAGES/background.png', 1)
            cv2imagex1 = cv2.cvtColor(load, cv2.COLOR_BGR2RGBA)
            load = Image.fromarray(cv2imagex1)
            load = load.resize((int(screen_widthx), int(screen_heightx)), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = tk.Label(image=render)
            img.image = render
            img.place(x=-1, y=0)

            

            cap = cv2.VideoCapture(0)

            def video_stream():
                if (cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, frame = cap.read()
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                im5 = img.resize((int(screen_widthx)-100, int(screen_heightx)-100), Image.ANTIALIAS)
                imgtk = ImageTk.PhotoImage(image=im5)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                lmain.after(1, video_stream)

            lmain = tk.Label(win, font=('Arial', int(15 * 3.5), 'bold'), fg="black", bg='white',anchor='center')
            lmain.place(x=50, y=50)
            video_stream()


            def start_recording():


                pass


            def stop_recording():

                pass



            style = Style()
            style.configure('TButton', font=
            ('calibri', 20, 'bold'),
                            borderwidth='4')

            # Changes will be reflected
            # by the movement of mouse.
            style.map('TButton', foreground=[('active', '!disabled', 'green')],
                      background=[('active', 'black')])

            self.b3 = ttk.Button(win, width=20, command=start_recording)
            self.b3.place(x=1320, y=50, width=45, height=50)

            self.b4 = ttk.Button(win, width=20, command=stop_recording)
            self.b4.place(x=1320, y=150, width=45, height=50)




        def store_INI(self):
            exit(0)



    window1 = Tk()
    window1.iconbitmap(default='IMAGES/home.ico')
    option_window = INIX(window1)
    window1.attributes('-fullscreen',TRUE)
    window1.config(background='white')
    window1.attributes('-alpha', 1)
    window1.title('PHOENIX-SOCIAL DISTANCING AND MASK DETECTOR' + version)
    window1.mainloop()

