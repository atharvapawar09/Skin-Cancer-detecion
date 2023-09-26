import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import CNN_ModelTrain_CPU as Model_frm
#import tfModel_test as tf_test
global fn,img,img2,img3
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Skin Disease Detection System")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('2.jpeg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)
#
lbl = tk.Label(root, text="Skin Cancer Detection System ", font=('times', 35,' bold '), height=1, width=25,bg="#FAEBD7",fg="blue")
lbl.place(x=300, y=0)






frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '),bg="tomato")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=20, y=100)


################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

def test_model_proc(fn):
    from keras.models import load_model
    #from keras.optimizers import Adam
    # from tensorflow.keras.optimizers import Adam


#    global fn
    IMAGE_SIZE = 227
    LEARN_RATE = 1.0e-4
    CH=3
    print(fn)
    if fn!="":
        # Model Architecture and Compilation
       
        model = load_model('model/best_model.h5')
            
        # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,CH)
            
        prediction = model.predict(img)
        print(np.argmax(prediction))
        Skin_diseases=np.argmax(prediction)
        
        if Skin_diseases==0:
            Cd="Benign"
            file = open(r"Report.txt", 'w')
            file.write("-----Patient Report-----\n As per input data and system model SkinCancer Detected for Respective Paptient."
                       "\n\n***Kindly Follow Medicatins***"
                    
                    )
        elif Skin_diseases==1:
            Cd="Malignant"
            file = open(r"Report.txt", 'w')
            file.write("-----Patient Report-----\n As per input data and system model No SkinCancer Detected for Respective Paptient."
                      "\n\n***Relax and Follow below mentioned Life Style to be Healthy as You Are!!!***"
                   
                   )
            
    
        A = Cd 
        
        return A
    
        

def train_model():
    
    update_label("Model Training Start...............")
    
    start = time.time()

    X=Model_frm.main()
    
    end = time.time()
        
    ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
    msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

    update_label(msg)

def test_model():
    global fn
    if fn!="":
        update_label("Model Testing Start...............")
        
        start = time.time()
    
        X=test_model_proc(fn)
        
        X1="Selected Image is {0}".format(X)
        
        end = time.time()
            
        ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
        msg="Image Testing Completed.."+'\n'+ X1 + '\n'+ ET
        fn=""
    else:
        msg="Please Select Image For Prediction...."
        
    update_label(msg)
    
    
def openimage():
   
    global fn
   
    fileName = askopenfilename(initialdir='/dataset', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=300
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])


#
#        gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)
#
#        gs = cv2.resize(gs, (x1, y1))
#
#        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250)
    #result_label1.place(x=300, y=100)
    img.image = imgtk
    img.place(x=300, y=100)
#        out_label.config(text=imgpath)

def convert_grey():
    global fn    
    IMAGE_SIZE=300
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    img2 = tk.Label(root, image=imgtk, height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=560, y=100)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=820, y=100)






def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=450)
    
    button4 = tk.Button(root, text="Display Histogram", command=graph,width=15, height=1,bg="red",fg="black", font=('times', 15, ' bold '))
    button4.place(x=550, y=570)


def graph():
    
    image3 =Image.open('accuracy.png')
    background_image1=ImageTk.PhotoImage(image3)
    background_label1 = tk.Label(root, image=background_image1)
    background_label1.image = background_image1
    background_label1.place(x=300, y=100)
    
    image4 =Image.open('loss.png')
    background_image2=ImageTk.PhotoImage(image4)
    background_label2 = tk.Label(root, image=background_image2)
    background_label2.image = background_image2
    background_label2.place(x=700, y=100)


    








#################################################################################################################
def window():
    root.destroy()




button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=10, y=50)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button2.place(x=10, y=120)

#button3 = tk.Button(frame_alpr, text="Train Model", command=train_model, width=12, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
#button3.place(x=10, y=160)
#
button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button4.place(x=10, y=190)
#
#
#button5 = tk.Button(frame_alpr, text="button5", command=window,width=8, height=1, font=('times', 15, ' bold '),bg="yellow4",fg="white")
#button5.place(x=450, y=20)


exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=10, y=260)



root.mainloop()