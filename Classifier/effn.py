import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import cv2
import os
import random
import pandas as pd
from random import shuffle
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
import seaborn as sns
from tensorflow.keras.models import Model
from PIL import Image
from scipy.ndimage.interpolation import rotate
import pickle
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import albumentations as A
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score,roc_auc_score,confusion_matrix
random.seed(0)
tf.compat.v1.disable_eager_execution()

root_dir = '/workstation/raid/home/p170059cs/data_from_b170007ec/Programs/Bhanu/CERVICAL2.0/Classifier/Cervix/ROIS/'
df = pd.read_csv("/workstation/raid/home/p170059cs/data_from_b170007ec/Programs/Bhanu/CERVICAL2.0/Classifier/Cervix/100_aug/Dataset_100_aug.csv", usecols=range(1,6))
df = df[df["Data_type"] == "Additional"]
df = df[df["Predicted_label"] == 0]
cleaned_images = list(df["Path"])

for i in range(len(cleaned_images)):
    img_dir = cleaned_images[i]
    cleaned_images[i] = img_dir.split('/')[-2] + '_'+img_dir.split('/')[-1].split('.')[0] + '.jpg'

img_dirs_temp = []
labels_temp = []
for file in sorted(os.listdir(root_dir))[1:]:
    if file in cleaned_images:
        file_dir = os.path.join(root_dir,file)
        img_dirs_temp.append(file_dir)
        file_list = file.split('_')
        try :
            labels_temp.append(int(file_list[-2])-1)
        except ValueError:
            labels_temp.append(int(file_list[-3])-1)

temp = list(zip(img_dirs_temp, labels_temp))
random.shuffle(temp)
img_dirs, labels = zip(*temp)
print(len(img_dirs),len(labels))

INPUT_SIZE = 512
BATCH_SIZE = 10
VALID_SPLIT = 0.1
EPOCHS = 100
ALPHA = 2.0
GAMMA = 3.0
LR = 0.0001

def metric(y_true,y_pred):
    log = -y_true*K.log(y_pred)
    log = K.sum(log,axis=-1)
    log = K.mean(log,axis=0)
    return log

def aug(max_angle=90):
    a = A.Compose([A.Rotate(limit = max_angle)])
    return a
#generator
class Generator(Sequence):
    def __init__(self,data,batch_size=BATCH_SIZE,input_size=INPUT_SIZE,is_train = True):
        self.img_dirs = data[0]
        self.labels = data[1]
        self.batch_size = batch_size
        self.input_size = input_size
        self.is_train = is_train
        if self.is_train:
            self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.img_dirs)/float(self.batch_size)))
    def on_epoch_end(self):
        if self.is_train:
            temp = list(zip(self.img_dirs,self.labels)) 
            shuffle(temp) 
            self.img_dirs, self.labels = zip(*temp)
    def __getitem__(self,idx):
        train_x = self.img_dirs[self.batch_size*idx:self.batch_size*(idx+1)]
        train_y =  self.labels[self.batch_size*idx:self.batch_size*(idx+1)]
        return self.generate(train_x,train_y)
    def generate(self,train_x,train_y):
        X = []
        Y = []
        for i,img in enumerate(train_x):
            y = np.zeros(3)
            y[train_y[i]] = 1
            if img.split('.')[-1]=='npy':
                roi = np.load(img)
            else:
                roi = cv2.imread(img)
                   
            #image = cv2.resize(image,(self.input_size,self.input_size))
            roi = cv2.resize(roi,(self.input_size,self.input_size))
            #img = np.concatenate([roi,image],axis=-1)
            X.append(roi/255)
            if self.is_train:
                data = {'image':roi}
                roi = aug()(**data)
                roi = roi['image']
                X.append(roi/255)
                Y.append(y)
            Y.append(y)
        return np.asarray(X), np.asarray(Y)

class prediction(Sequence):
    def __init__(self,data,batch_size=BATCH_SIZE,input_size=INPUT_SIZE):
        self.img_dirs = data
        self.batch_size = batch_size
        self.input_size = input_size
    def __len__(self):
        return int(np.ceil(len(self.img_dirs)/float(self.batch_size)))
    def on_epoch_end(self):
            pass
    def __getitem__(self,idx):
        train_x = self.img_dirs[self.batch_size*idx:self.batch_size*(idx+1)]
        return self.generate(train_x)
    def generate(self,train_x):
        X = []
        for i,img in enumerate(train_x):
            if img.split('.')[-1]=='npy':
                roi = np.load(img)
            else:
                roi = cv2.imread(img)   
            #image = cv2.resize(image,(self.input_size,self.input_size))
            roi = cv2.resize(roi,(self.input_size,self.input_size))
            #img = np.concatenate([roi,image],axis=-1)
            X.append(roi/255)
        return np.asarray(X)
def focal_loss(alpha,gamma):
    def loss_fn(y_true,y_pred):
        y_pred = K.clip(y_pred,1e-5,1-1e-5)
        loss = alpha*((1-y_pred)**gamma)*y_true*K.log(y_pred)
        loss = -K.sum(loss,axis=-1)
        return loss
    return loss_fn

#supporting blocks
def up_image(input_1,c=None):
    x = UpSampling2D((2,2))(input_1)
    x_ = Conv2D(c//4,(1,1),kernel_initializer='glorot_uniform',activation='relu')(x)
    return x

def build_model(ALPHA,GAMMA):
    encoder = efn.EfficientNetB4(include_top=False,input_shape = (INPUT_SIZE,INPUT_SIZE,3))    
    #x_ = up_image(encoder.output,c = 256)
    #x_ = up_image(x_,c = 128)
    #x_ = up_image(x_,c = 64)
    #x_ = Conv2D(32,(2,2),kernel_initializer = 'glorot_uniform',padding='same',activation='relu')(x_)
    #x_ = Conv2D(3,(1,1),kernel_initializer='glorot_uniform',padding='same',activation='sigmoid')(x_)
    #arbi = Model(encoder.input,x_)
    #arbi.load_weights('/home/b170007ec/Programs/Manoj/DAE/model2_dae.h5')
    x = encoder.output
    x = Dropout(0.5)(x)
    x = Conv2D(512,(1,1),padding = 'same',kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(256,(3,3),padding = 'same',kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Conv2D(128,(1,1),padding = 'same',kernel_initializer = 'glorot_uniform',activation='relu')(x)
    x = MaxPool2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(3,activation='softmax')(x)
    model = Model(encoder.input,x)
    model.compile(optimizer=Adam(lr=LR),loss=focal_loss(ALPHA,GAMMA),metrics = ['acc',metric])
    return model

weights = []
SEED = 10
img_dirs = np.asarray(img_dirs)
labels = np.asarray(labels)
train_dirs, val_dirs, train_labels, val_labels = train_test_split(img_dirs, labels,
                                                    stratify=labels, 
                                                    test_size=0.2)
print(len(train_dirs), len(val_dirs))
train_gen = Generator((train_dirs,train_labels),is_train=True)
val_gen = Generator((val_dirs,val_labels),is_train=False)
STEPS_PER_EPOCH = len(train_dirs)//BATCH_SIZE
model = build_model(ALPHA,GAMMA)
name = '/workstation/raid/home/p170059cs/data_from_b170007ec/Programs/Bhanu/CERVICAL2.0/Classifier/Cervix/100_aug/models/'+'model'+'-{val_acc:03f}'+'.h5'
early_stopping = EarlyStopping(monitor = 'val_acc', min_delta=0, patience = 6, verbose = 1)
checkpoint = ModelCheckpoint(name,monitor = 'val_acc', save_best_only = True, verbose = 1, period = 1,mode='max')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=1, epsilon=LR, mode='max')  
history = model.fit_generator(train_gen,steps_per_epoch=STEPS_PER_EPOCH,validation_data = val_gen,epochs = EPOCHS,callbacks = [checkpoint,reduce_lr_loss,early_stopping])
f = os.listdir("/workstation/raid/home/p170059cs/data_from_b170007ec/Programs/Bhanu/CERVICAL2.0/Classifier/Cervix/100_aug/models/")
f = sorted(f)
weights.append("/workstation/raid/home/p170059cs/data_from_b170007ec/Programs/Bhanu/CERVICAL2.0/Classifier/Cervix/100_aug/models/"+str(f[-1]))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("accuracy.jpg")
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss.jpg")
plt.clf()

root_dir = '/workstation/raid/home/p170059cs/data_from_b170007ec/Programs/Bhanu/CERVICAL2.0/Classifier/Cervix/ROIS/'
df = pd.read_csv("/workstation/raid/home/p170059cs/data_from_b170007ec/Programs/Bhanu/CERVICAL2.0/Dataset/Cervical Data.csv", usecols=range(1,5))
df["Noise"].replace({"0": 0, "1": 1, "Nan": 2}, inplace=True)
df = df[df["Data_type"] == "Train"]
df = df[df["Noise"] == 0]
cleaned_images = list(df["Path"])
for i in range(len(cleaned_images)):
    img_dir = cleaned_images[i]
    cleaned_images[i] = img_dir.split('/')[-2] + '_'+img_dir.split('/')[-1].split('.')[0] + '.jpg'

test_dirs = []
test_labels = []
for file in sorted(os.listdir(root_dir))[1:]:
    if file in cleaned_images:
        file_dir = os.path.join(root_dir,file)
        test_dirs.append(file_dir)
        file_list = file.split('_')
        try :
            test_labels.append(int(file_list[-2])-1)
        except ValueError:
            test_labels.append(int(file_list[-3])-1)

pred_gen = prediction(test_dirs)
def test(model,weights):
    predictions = np.zeros((len(test_dirs),3))
    for i,weight in enumerate(weights):
        model.load_weights(weight)
        print("weights loaded...")
        predictions = predictions+model.predict_generator(pred_gen,verbose=1)
    pred_labels = []
    for i in range(predictions.shape[0]):
        pred_labels.append(np.argmax(predictions[i,:]))
    f = open("results.txt",'w+')
    f.write("accuracy:{}\n".format(accuracy_score(test_labels,pred_labels)))
    f.write("precision_score:{}\n".format(precision_score(test_labels,pred_labels,average=None)))
    f.write("recal_score:{}\n".format(recall_score(test_labels,pred_labels,average=None)))
    f.write("f1_score:{}\n".format(f1_score(test_labels,pred_labels,average=None)))
    f.close()
    matrix = confusion_matrix(test_labels,pred_labels)
    ax = plt.subplot()
    sns.heatmap(matrix,annot=True,ax = ax)
    ax.set_xlabel("prediction labels")
    ax.set_ylabel("true labels")
    plt.savefig("rconfusion matrix.png")
model = build_model(ALPHA,GAMMA)
test(model,weights)
