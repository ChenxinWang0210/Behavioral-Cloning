import csv
import json
import cv2
from scipy import ndimage
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.misc import imresize
import pydot


###### Load Udacity data ###############
data_path ='/opt/carnd_p3/data/'
samples = []
with open(data_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        samples.append(line)
 
#### split train and validation samples
shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


    
    
### Resize Image for fast training ###
img_height = 70
img_width = 224
def resize_img(img):
    crop_img = img[53:153,:,:]
    resized_img=imresize(crop_img, (img_height, img_width), interp='bilinear')
   
    return resized_img

### Image Augmentation ###########

def augment_image(img):
    # random translate vertically
    cols = img.shape[1]
    rows = img.shape[0]
    transY = np.random.randint(-10,10,1)
    M = np.float32([[1,0,0],[0,1,transY]])
    img = cv2.warpAffine(img,M,(cols,rows))
    
    image = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    # random brightness
    random_bright = 1+np.random.uniform(-0.3,0.3)
    image[:,:,0] = image[:,:,0]*random_bright
    image[:,:,0][image[:,:,0]>255]  = 255
    
    # random shadow
    mid =np.random.randint(0,rows)
    shadow_factor = np.random.uniform(0.7, 0.9)
    if np.random.randint(2)==0:
        image[:,0:mid,0] = image[:,0:mid,0]*shadow_factor
    else:
        image[:,mid:,0] = image[:,mid:,0]*shadow_factor
        
    image = cv2.cvtColor(image,cv2.COLOR_YUV2RGB)
    return image    
    
       
   
# create adjusted steering measurements for the side camera images
correction = 0.25 # this is a parameter to tune

### Python generator function
def generator(samples, batch_size=64, valid_flag=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = ndimage.imread(data_path+'IMG/'+batch_sample[0].split('/')[-1])
                center_image = resize_img(center_image)
                if not valid_flag:
                    center_image = augment_image(center_image)
            
                center_angle = float(batch_sample[3])
                # Append center image to the dataset
                images.append(center_image)
                angles.append(center_angle)
                
                # Append flipped center image to the dataset only  if the steering angle>0.3
                if abs(center_angle)>0.3:
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
                # Append left image to the dataset 
                left_image = ndimage.imread(data_path+'IMG/'+batch_sample[1].split('/')[-1])
                left_image = resize_img(left_image)
                if not valid_flag:
                    left_image = augment_image(left_image)
                left_angle=center_angle+correction
                images.append(left_image)
                angles.append(left_angle)
                # Append flipped left image to the dataset only  if the steering angle>0.3
                if abs(left_angle)>0.3:
                    images.append(np.fliplr(left_image))
                    angles.append(-left_angle)
                    
                # Append right image to the dataset 
                right_image = ndimage.imread(data_path+'IMG/'+batch_sample[2].split('/')[-1])
                right_image = resize_img(right_image)
                if not valid_flag:
                    right_image = augment_image(right_image)
                right_angle=center_angle-correction
                images.append(right_image)
                angles.append(right_angle)
                # Append flipped right image to the dataset only  if the steering angle>0.3
                if abs(right_angle)>0.3:
                    images.append(np.fliplr(right_image))
                    angles.append(-right_angle)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
# using the generator function on both training and validation set
batch_size =32
train_generator = generator(train_samples, batch_size,valid_flag=False)
validation_generator = generator(validation_samples, batch_size,valid_flag=True)

# Import keras functions
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D 
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils.vis_utils import plot_model

fine_tune_model=False
if fine_tune_model:
    ### Fine tune model ##############################
    print('##### Training CNN in fine tune mode #####')
    model=load_model('model.h5')
    model.compile(loss='mse', optimizer='adam')
    
else:
######### Lenet Neural Network (Underfitting) #######################
#     model = Sequential()
#     model.add(Lambda(lambda x: x/255.0-0.5, input_shape =(img_height,img_width,3)))
#     ## Convolutional Layer 1 
#     model.add(Conv2D(6,kernel_size = (5,5), strides=(1,1),padding='valid',activation='relu'))
#     ## Pooling Layer
#     model.add(MaxPooling2D((2,2)))
#     ## Convolutional Layer 2
#     model.add(Conv2D(16,kernel_size = (5,5), strides=(1,1),padding='valid',activation='relu'))
#     ## Pooling Layer
#     model.add(MaxPooling2D((2,2)))
#     model.add(Flatten())
#     ## Full-connected Layer 1
#     model.add(Dense(120,activation='relu'))
#     ## Full-connected Layer 2
#     model.add(Dense(84,activation='relu'))
#     ## Output Layer 
#     model.add(Dense(1))
################################################
    
###### Modified NVIDIA neural network ##########
    model = Sequential()
    ## Normalization Layer
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape =(img_height,img_width,3)))
#     ## Cropping Images
#     model.add(Cropping2D(cropping=((70,22),(0,0))))
    ## Convolutional Layer 1 
    model.add(Conv2D(24,kernel_size = (5,5), strides=(1,1),padding='valid',activation='relu'))
    ## Pooling Layer
    model.add(MaxPooling2D((2,2)))
    ## Convolutional Layer 2
    model.add(Conv2D(36,kernel_size = (5,5), strides=(1,1),padding='valid',activation='relu'))
    ## Pooling Layer
    model.add(MaxPooling2D((2,2)))
    ## Convolutional Layer 3
    model.add(Conv2D(48,kernel_size = (5,5), strides=(1,1),padding='valid',activation='relu'))
    ## Convolutional Layer 4
    model.add(Conv2D(64,kernel_size = (5,5), strides=(1,1),padding='valid',activation='relu'))
    ## Convolutional Layer 5
    model.add(Conv2D(64,kernel_size = (3,3), strides=(1,1),padding='valid',activation='relu'))
    ## Convolutional Layer 6
    model.add(Conv2D(64,kernel_size = (3,3), strides=(1,1),padding='valid',activation='relu'))
    model.add(Flatten())
    ## Full-connected Layer 1
    model.add(Dense(100,activation='relu'))
    ## Full-connected Layer 2
    model.add(Dense(50,activation='relu'))
    ## Full-connected Layer 3
    model.add(Dense(10,activation='relu'))
    ## Output Layer 
    model.add(Dense(1))
    
    
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
print(model.summary())   

model.compile(loss='mse', optimizer='adam')

### Save the best model 
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

### Stop training when val_loss has stopped improving
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')

### Stream epoch results to a csv file
csv_logger = CSVLogger('training.log')


model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//batch_size, \
                    validation_data=validation_generator,validation_steps=len(validation_samples)//batch_size, \
                    nb_epoch=8,verbose=1,callbacks = [checkpoint,earlystop,csv_logger])




# save the model and weights
model_json = model.to_json()
with open("model.json","w") as json_file:
    json.dump(model_json,json_file)
model.save_weights('model_weights.h5')

    

