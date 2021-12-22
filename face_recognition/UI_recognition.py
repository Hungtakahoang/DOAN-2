import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import numpy
import cv2

# loading model
model = load_model('my_model1 (89%).h5')

classes = np.array(['Albert Einstein', 'Cristiano Ronaldo', 'Donald Trump', 'Galileo Galilei', 'Joe Biden', 'Lionel Messi', 
                    'Mark Zuckerberg', 'Nicola Tesla', 'Son Tung MTP', 'Tran Thanh'])

#initialise GUI
root=tk.Tk()
root.geometry('800x573')
root.title('Face Recognition')
bgImage = ImageTk.PhotoImage(Image.open("background2.jpg"))
background_label = Label(root, image=bgImage)
background_label.place(x=0, y=20, relheight=1, relwidth=1)

label=Label(root,background='white', font=('arial',15,'bold'))
sign_image = Label(root)




def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((224,224))
    image = numpy.expand_dims(image, axis=0) #(Batch_size, height, width, channel)
    image = numpy.array(image)
    pred = model.predict(image)[0] # [0, 0, 0 ,0 , 1, 0, 0, 0, 0, 0]
    index = np.argmax(pred)
    face = classes[index]  
    print(face)
    label.configure(foreground='#011638', text=face) 

def show_classify_button(file_path):
    classify_b=Button(root,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5, relief='raised', overrelief='groove', cursor='cross',
                    activebackground="red", activeforeground="white")
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.51)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((root.winfo_width()/2.25),(root.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

# instruction 
instruction = tk.Label(root, text="Select Image file on your computer")
instruction.place(relx=0.34,rely=0.93)
instruction.configure(background = '#061b3a', foreground='white',font=('Raleway', 13, "italic"))
upload=Button(root,text="Browse",command=upload_image,padx=10,pady=5, relief='raised', overrelief='groove', cursor='cross',
                activebackground="red", activeforeground="white")
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.place(relx=0.30, rely=0.90)

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(root, text="Face Recognition",pady=20, padx=800, font=('arial',20,'bold'))
heading.configure(background='#030f1b',foreground='white')
heading.pack()
root.mainloop()
