from PIL import ImageTk
import PIL.Image
from tkinter import *
import tkinter as tk
from predictionFun import predcition
window=Tk()

def qf():
    text=User_input.get()
    res=predcition(text)
    lb2.config(text=str(res))
    logo = ImageTk.PhotoImage(PIL.Image.open(text))
    w1=Label(window, image=logo)
    w1.config(image=logo)
    w1.image = logo
    window.update_idletasks()

path = 'bg6-r.jpg'
logo = ImageTk.PhotoImage(PIL.Image.open(path))
w1 = Label(window, image=logo).pack(side="left")


lbl=Label(window, text="Healthy Meal System", fg='black', font=("Helvetica 25 bold"))
lbl.place(x=200, y=50)# العنوان

lbl=Label(window, text="enter image:", fg='black', font=("Helvetica 15 bold"))
lbl.place(x=200, y=231)#ليبل

#path = 'bg6-r.jpg>'
#logo = ImageTk.PhotoImage(PIL.Image.open(path))
#w1 = Label(window, image=logo).pack(side="left")

User_input = tk.Entry(window,width =50)# input from user
User_input.place(x=350, y=240)




btn=tk.Button(window, text="predict", height="2", width="20", bg='pink',fg='white',command=qf)
btn.place(x=400, y=300)



#lbl=Label(window, text="Your Meal IS :", fg='black', font=("Helvetica 15 bold "))
#lbl.place(x=450, y=350)
lb2=Label(window,  text="Meal", fg='black', font=("Helvetica 15 bold "))
lb2.place(x=500, y=400)


window.title('healthy meal system')
window.geometry("730x600+10+20")

window.mainloop()