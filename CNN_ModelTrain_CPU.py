#Import some packages to use
import numpy as np

#To see our directory
import os
import gc   #Gabage collector for cleaning deleted data from memory
from PIL import Image




def main():
    basepath="D:/new project/SkinCancer/SkinCancer"
    #=================================================================================
#    CSV_FILE=r"E:\Alka_python_2019_20\Melanoma Recognition\Main_Project\dataset\skinDlabel.csv"
    #=================================================================================
    train_dir=r"D:\\100% code\\ project\\SkinCancer\\SkinCancer\\dataset\\train"
    test_dir=r"D:\\100% code\\ project\\SkinCancer\\SkinCancer\\dataset\\testing"
    
    print("Train Images " + str(len(os.listdir(train_dir))))
    print("Test Images " + str(len(os.listdir(test_dir))))
        
    imlist = os.listdir(train_dir)
   
     # create matrix to store all flattened images
   
    immatrix = np.array([np.array(Image.open(train_dir + '\\' + im2)).flatten()
                  for im2 in imlist],'f')

#    a=0
#    for file in imlist:
#        a=a+1 
#        im = Image.open(train_dir + '\\' + file)       
#        img = im.resize((227,227))
#        img.save(train_dir + '\\' + file)
#        print(a)
    
#    del train_imgs
    gc.collect()
    
    #=================================================================================
#    #Lets declare our image dimensions
#    #we are using coloured images. 
#    import csv
#    
#    def csv_search(img_name):
#    
#        csv_file = csv.reader(open(CSV_FILE, "r"), delimiter=",")
#    
#        # loop through csv list
#        for row in csv_file:
#            if img_name == str(row[2])+'.jpeg':
#                return row[0]
#    #A function to read and process the images to an acceptable format for our model
#    def read_and_process_image(list_of_images):
#        y=[]
#        for image in list_of_images:
#            Fname=image.split('/')
#            Fint=int(len(Fname))-1
#            cls_N=csv_search(Fname[Fint])
#            if int(cls_N)==0:
#                y.append(0)
#            elif int(cls_N)==1:
#                y.append(1)
#        return y
#    
    
    ####=================================================================================
    #get the train and label data
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    from keras.utils import np_utils
    
    img_cols=227
    img_rows=227
    
    listing = os.listdir(train_dir)
    num_samples= np.size(listing)

#    y = read_and_process_image(train_imgs)
#    y=np.array(y,dtype = int)
    
    label=np.ones((num_samples,),dtype = int)
    
    label[0:952]=0
    label[952:]=1
   

    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
        
    nb_classes = 2
    
    (X, y) = (train_data[0],train_data[1])
    
    
    # STEP 1: split X and y into training and testing sets
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=4)
    
    X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 3)
    X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 3)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_test /= 255
    
    print('y_train shape:', Y_train.shape)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    
    
    ####=================================================================================
    from keras.optimizers import Adam
    from tensorflow.keras.optimizers import Adam
    
    from Alexnet import AlexNet
    
    # Variables
    CLASSES = 2
    IMAGE_SIZE = 227
    CHANNELS = 3
    NUM_EPOCH = 1
    LEARN_RATE = 1.0e-4
    BATCH_SIZE = 64
    MODEL_NAME="best_model.h5"

    # Model Architecture and Compilation
    model = AlexNet(CLASSES, IMAGE_SIZE, CHANNELS)
    adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    from keras.callbacks import ModelCheckpoint
    
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='auto')
    
    steps_per_epoch = int(len(Y_train) / BATCH_SIZE)  # 300
#    validation_steps = int(len(Y_test) / BATCH_SIZE)  # 90
    
    callbacks_list = [checkpoint]
#    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
    # Training
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                           batch_size=BATCH_SIZE,shuffle="batch", epochs=NUM_EPOCH,callbacks=callbacks_list,verbose=1)
    
    #=======================================================================================
    # import matplotlib.pyplot as plt
    # # summarize history for accuracy
    # plt.plot(model.history['accuracy'])
    # plt.plot(model.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(basepath + "/accuracy.png",bbox_inches='tight')

    # plt.show()
    # # summarize history for loss
    
    # plt.plot(model.history['loss'])
    # plt.plot(model.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(basepath + "/loss.png",bbox_inches='tight')
    
    # plt.show()
    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    
   
    

    # plt.show() ###=================================================================================
    # print("Model Saved")
    ###=================================================================================
    
    predictions = model.predict(X_train)
    accuracy = 0
    for prediction, actual in zip(predictions, Y_train):
        predicted_class = np.argmax(prediction)
        actual_class = np.argmax(actual)
        if(predicted_class == actual_class):
            accuracy+=1
    
    accuracy =( accuracy / len(Y_train))*100
    
    A = "Training Accuracy is {0}".format(accuracy)
    
    
    predictions = model.predict(X_test)
    accuracy = 0
    for prediction, actual in zip(predictions, Y_test):
        predicted_class = np.argmax(prediction)
        actual_class = np.argmax(actual)
        if(predicted_class == actual_class):
            accuracy+=1
    
    accuracy = (accuracy / len(Y_test))*100
    
    B = "Testing Accuracy is {0}".format(accuracy)
    
    model.s
    (MODEL_NAME)
    
    msg=A+'\n'+B+'\n'+ "Saved as " + MODEL_NAME

    
    return msg

if __name__=='__main__' : main()