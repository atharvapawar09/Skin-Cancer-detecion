# Building architecture of our CNN classifier
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# import BatchNormalization
#from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout


from tensorflow.keras.layers import BatchNormalization

def AlexNet(CLASSES, IMAGE_SIZE, CHANNELS):
    
    classifier = Sequential()

    # First Convolution Block
    classifier.add(Conv2D(
        96, 11, strides=(4, 4), input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    classifier.add(BatchNormalization())

    # Second Convolution Block
    classifier.add(Conv2D(256, 5, strides=(1, 1),
                          activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    classifier.add(BatchNormalization())

    # Third Convolution Block
    classifier.add(Conv2D(384, 3, strides=(1, 1),
                          activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Fourth Convolution Block
    classifier.add(Conv2D(384, 3, strides=(1, 1),
                          activation='relu', padding='same'))

    # Fifth Convolution Block
    classifier.add(Conv2D(256, 3, strides=(1, 1),
                          activation='relu', padding='same'))
    classifier.add(Dropout(0.5))

    # Fully connected layer
    classifier.add(Flatten())

    # First hidden unit
    classifier.add(Dense(512, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(0.5))

    # Second hidden unit
    classifier.add(Dense(512, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(0.5))

    # Output layer
    classifier.add(Dense(CLASSES, activation='softmax',
                         kernel_initializer='uniform'))

    classifier.summary()
    
    return classifier