import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np


import numpy
#load the trained model to classify sign
from tensorflow.keras.models import load_model
model = load_model('my_model1 (89%).h5')

#dictionary to label all traffic signs class.
# classes = { 0:'Albert Einstein',
#             1:'Alexander Fleming', 
#             2:'Dmitri Mendeleev', 
#             3:'Galileo Galilei', 
#             4:'Issac Newton', 
#             5:'Marie Curie', 
#             6:'Michael Faraday', 
#             7:'Nicola Tesla', 
#             8:'Otto Hahn', 
#             9:'Thomas Edison'
#             }
classes = np.array(['Albert Einstein', 'Alexander Fleming', 'Dmitri Mendeleev', 'Galileo Galilei', 'Issac Newton', 'Marie Curie', 
                    'Michael Faraday', 'Nicola Tesla', 'Otto Hahn', 'Thomas Edison'])

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Face Recognition')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((200,200))
    image = numpy.expand_dims(image, axis=0)
    print(image.shape)
    image = numpy.array(image)
    pred = model.predict(image)[0]
    index = np.argmax(pred)
    face = classes[index]  
    print(face)
    label.configure(foreground='#011638', text=face) 

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Face Recognition",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()