
import shutil

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
# from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib

from sklearn.metrics import mean_absolute_error as mae
from math import sqrt
from sklearn.metrics import mean_squared_error

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

############## Preprocessing ############

def retrievePixels(path):
    img = tf.keras.utils.load_img(path, grayscale=False, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img).reshape(1, -1)[0]
    return x

#CFD
# Target refers to the folder name
# Some folders include several images. I will append list of including file names as a column.
cfd_df_raw = pd.read_csv(r"C:\Users\nunok\PycharmProjects\pythonProject2\cfd\CFD Version 3.0\metadata.csv")


# Target refers to the folder name
# Some folders include several images. I will append list of including file names as a column.
def getFileNames(target):
    files = []
    file_count = 0
    path = r"C:\Users\nunok\PycharmProjects\pythonProject2\cfd\CFD Version 3.0\Images\%s" % (target)
    for r, d, f in os.walk(path):
        for file in f:
            if ('.jpg' in file) or ('.jpeg' in file) or ('.png' in file) :
                files.append(file)
    return files


cfd_df_raw["files"] = cfd_df_raw.Target.apply(getFileNames)

# I should store each file an instance. In other words, number of rows of the data set should be equal to the number of total files.
# I will build two for loops. First iterates on the folder names (Target column value) and second iterates on the files.
# store in a new df
cfd_instances = []
for index, instance in cfd_df_raw.iterrows():
    folder = instance.Target
    score = instance['Score']
    for file in instance.files:
        if file[-5] == "N":
            tmp_instance = []
            tmp_instance.append(folder)
            tmp_instance.append(file)
            tmp_instance.append(score)
            cfd_instances.append(tmp_instance)

df_cfd = pd.DataFrame(cfd_instances, columns=["folder", "file", "Score"])

df_cfd['exact_file'] = "C:\\Users\\nunok\\PycharmProjects\\pythonProject2\\cfd\\CFD Version 3.0\\Images"+ "\\" + df_cfd["folder"]+ "\\" +df_cfd['file']
df_cfd['pixels'] = df_cfd['exact_file'].apply(retrievePixels)
#normalize
df_cfd["Score"] = df_cfd["Score"]/7

shuffle = df_cfd.sample(frac = 1, random_state = 42)
shuffle.reset_index(drop=True)

train_df_cfd = shuffle.iloc[0:358]
val_df_cfd = shuffle.iloc[359:478]
test_df_cfd = shuffle.iloc[479:]


# read csv Preprocess photos
#SCUT-FBP dataset
photos = pd.read_csv(r"C:\Users\nunok\PycharmProjects\pythonProject2\SCUT-FBP5500_v2\train_test_files\All_labels.csv")
photos['exact_file'] = "C:\\Users\\nunok\\PycharmProjects\\pythonProject2\\SCUT-FBP5500_v2\\Images\\" + photos["Target"]
photos['pixels'] = photos['exact_file'].apply(retrievePixels)
#normalize
photos["Score"] = photos["Score"]/5

shuffle = photos.sample(frac = 1, random_state = 42)
shuffle = shuffle.sort_values(by=['Target'])
shuffle.reset_index(drop=True)

train_photos= shuffle.iloc[0:3300]
val_photos = shuffle.iloc[3301:4401]
test_photos = shuffle.iloc[4402:]


#KDEF
# Target refers to the folder name
# Some folders include several images. I will append list of including file names as a column. (not anymore)
kdef = pd.read_csv(r"C:\Users\nunok\PycharmProjects\pythonProject2\KDEF\metadata_kdef.csv")

#kdef['folder'] = kdef.apply(lambda x: x['Target'][:4],axis=1)
#kdef['exact_file'] = "C:\\Users\\nunok\\PycharmProjects\\pythonProject2\\KDEF\\" + kdef['folder'] + "\\" +kdef['Target'] + ".JPG"
kdef['exact_file'] = "C:\\Users\\nunok\\PycharmProjects\\pythonProject2\\KDEF\\" +kdef['Target'] + ".JPG"
kdef['pixels'] = kdef['exact_file'].apply(retrievePixels)

#normalize
kdef["Score"] = kdef["Score"]/7

#train/val/test

shuffle = kdef.sample(frac = 1, random_state = 42)
shuffle = shuffle.sort_values(by=['Target'])
shuffle.reset_index(drop=True)

train_kdef = shuffle.iloc[0:124]
val_kdef = shuffle.iloc[125:166]
test_kdef = shuffle.iloc[167:]

#Faces database
# Target refers to the folder name
# Some folders include several images. I will append list of including file names as a column.
faces = pd.read_csv(r"C:\Users\nunok\PycharmProjects\pythonProject2\FACES_ A database of facial expressions in younger, middle-aged, and older women and men\metadata_faces.csv")


faces['exact_file'] = "C:\\Users\\nunok\\PycharmProjects\\pythonProject2\\FACES_ A database of facial expressions in younger, middle-aged, and older women and men\\" + faces["Target"]
faces['pixels'] = faces['exact_file'].apply(retrievePixels)
#normalize
faces["Score"] = faces["Score"]/100

shuffle = faces.sample(frac = 1, random_state = 42)
shuffle = shuffle.sort_values(by=['Target'])
shuffle.reset_index(drop=True)

train_faces = shuffle.iloc[0:618]
val_faces = shuffle.iloc[619:823]
test_faces = shuffle.iloc[824:]


train = [train_df_cfd, train_faces, train_photos, train_kdef]
val = [val_df_cfd, val_faces, val_photos, val_kdef]
test = [test_df_cfd, test_faces, test_photos, test_kdef]

train = pd.concat(train)
val = pd.concat(val)
test = pd.concat(test)

#lista = [df_cfd,faces,photos,kdef]
#lista = [df_cfd,faces,kdef]
#photos = pd.concat(lista)



def features_preprocessing(x):
    features = []
    #pixels = photos['pixels'].values
    pixels = x['pixels'].values
    for i in range(0, pixels.shape[0]):
        features.append(pixels[i])

    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)

    features = features / 255
    return features


X_train = features_preprocessing(train)
y_train = train["Score"].values

X_val = features_preprocessing(val)
y_val = val["Score"].values

X_test = features_preprocessing(test)
y_test = test["Score"].values

#score = np.array(photos["Score"])

#X_train, X_test, y_train, y_test = train_test_split(features_preprocessing(), photos["Score"].values, test_size=0.40, random_state=42)
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

############## FIT ###########

import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, \
        Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential
from sklearn.metrics import mean_absolute_error as mae
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import matplotlib.pyplot as plt


class models_fit:
    def __init__(self, models):
        self.models = models
        self.histories = {}
        self.predictions = {}

    def fit2(self, lr=0.001, epochs=100, patience=30, verbose=0):
        model_dir = 'models'
        count = 0
        for key, value in self.models.items():
            model_name = key
            model_path = model_dir + '/' + model_name + '.h5'
            if not os.path.isdir(model_dir): os.mkdir(model_dir)

            basemodel = value

            model = Sequential(name=model_name)
            model.add(basemodel)
            model.add(Dense(1))


            model.layers[0].trainable = False
            model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

            checkpointer = ModelCheckpoint(filepath=model_name + ".hdf5"
                                           , monitor="val_loss", verbose=verbose
                                           , save_best_only=True, mode='auto'
                                           )

            earlyStop = EarlyStopping(monitor='val_loss', patience=patience)

            history = model.fit(X_train, y_train, epochs=epochs
                                , validation_data=(X_val, y_val), callbacks=[checkpointer, earlyStop]
                                )
            # validation loss
            self.histories[key] = history

            # predictions
            predictions = model.predict(X_test)

            self.predictions[model_name] = predictions

    #fine-tuning
    def fit(self, lr=0.001, epochs=100, patience=30, verbose=0,fine_tuning = False,n_layers = 2):
        model_dir = 'models'
        count = 0
        for key, value in self.models.items():
            model_name = key
            model_path = model_dir + '/' + model_name + '.h5'
            if not os.path.isdir(model_dir): os.mkdir(model_dir)

            basemodel = value
            basemodel_layer_list = basemodel.layers

            model = Sequential(name=model_name)

            for i in range(len(basemodel_layer_list) - 1):
                model.add(basemodel_layer_list[i])

            if fine_tuning == False:
                if n_layers >0:
                    for layers in model.layers[:-(n_layers)]:
                        layers.trainable = False
                else:
                    for layers in model.layers:
                        layers.trainable = False

            model.add(Dense(1))



            model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

            checkpointer = ModelCheckpoint(filepath=model_name + ".hdf5"
                                           , monitor="val_loss", verbose=verbose
                                           , save_best_only=True, mode='auto'
                                           )

            earlyStop = EarlyStopping(monitor='val_loss', patience=patience)

            history = model.fit(X_train, y_train, epochs=epochs
                                , validation_data=(X_val, y_val), callbacks=[checkpointer, earlyStop]
                                )
            # validation loss
            self.histories[key] = history

            # predictions
            predictions = model.predict(X_test)

            self.predictions[model_name] = predictions

    def predict(self):
        df_predictions = pd.DataFrame(y_test, columns=['actuals'])

        for key, value in self.predictions.items():
            df_predictions[key] = value

        return df_predictions

    def metrics(self):
        dici_metrics = {}
        actuals = self.predict()["actuals"]

        for model in self.models:
            ####temp dic
            temp = {}

            predictions = self.predict()[model]

            temp["person correlation"] = self.predict()[["actuals", model]].corr(method='pearson').values[
                0, 1]
            temp["mae"] = mae(actuals, predictions)
            temp["rmse"] = sqrt(mean_squared_error(actuals, predictions))

            dici_metrics[model] = temp
        return dici_metrics


############## Performance ############
def loss_plot(model_fit):
    best_iteration = np.argmin(model_fit.history['val_loss']) + 1
    val_scores = model_fit.history['val_loss'][0:best_iteration]
    train_scores = model_fit.history['loss'][0:best_iteration]

    plt.plot(val_scores, label='val_loss')
    plt.plot(train_scores, label='train_loss')
    plt.legend(loc='upper right')
    plt.show()


def regression_plot(predictions,actuals):
    min_limit = photos.Score.min()
    max_limit = photos.Score.max()
    best_predictions = []

    # for i in np.arange(1, 7, 0.01):
    for i in np.arange(int(min_limit), int(max_limit) + 1, 0.01):
        best_predictions.append(round(i, 2))

    plt.scatter(best_predictions, best_predictions, s=1, color='black', alpha=0.5)
    plt.scatter(predictions, actuals, s=20, alpha=0.1)


##############Prediction############

def getFileNames(target):
    files = []
    file_count = 0
    path = r"C:\Users\nunok\PycharmProjects\pythonProject2\cfd\CFD Version 3.0\fotos\predict\%s" % (target)
    for r, d, f in os.walk(path):
        for file in f:
            if ('.jpg' in file) or ('.jpeg' in file) or ('.png' in file) or ('.JPG' in file):
                files.append(file)
    return files


male_masculinized = []
male_feminized = []
female_masculinized = []
female_feminized = []
female_norm = []
male_norm = []
flat_list = []


def file_to_dic():
    #code to fem: 0
    #code to male: 1
    #if congruent(female feminized): 00; male masculinized:11; male_norm:111; male_incongruent:10


    gender = []
    congruent = []
    flat_list = []

    path = r"C:\Users\nunok\PycharmProjects\pythonProject2\predict"
    for r, f, d in os.walk(path):
        # folder name
        for folder in f:

            for r, file, d in os.walk(r"C:\Users\nunok\PycharmProjects\pythonProject2\predict\%s" % (folder)):
                # filename in folder
                for img in d:
                    flat_list.append(img)

                    if folder == "male":
                        if (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)) and "masc" in img:
                            gender.append("male")
                            congruent.append("11")
                        elif (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)) and "zed" in img:
                            gender.append("male")
                            congruent.append("10")

                        elif ('.jpg' in img) or ('.jpeg' in img) or ('.png' in img):
                            male_norm.append("male")
                            congruent.append("111")

                    elif folder == "female":
                        if (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)) and "masc" in img:
                            gender.append("female")
                            congruent.append("01")
                        elif (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)) and "zed" in img:
                            gender.append("female")
                            congruent.append("00")
                        elif (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)):
                            gender.append("female")
                            congruent.append("000")

    dic = {"gender": gender, "congruent": congruent,"flat_list":flat_list}
    return dic


def create_data():
    df = pd.DataFrame()
    df["file"] = file_to_dic()["flat_list"]
    df["gender"] = file_to_dic()["gender"]
    df["congruent"] = file_to_dic()["congruent"]


    df['exact_file'] = "C:\\Users\\nunok\\PycharmProjects\\pythonProject2\\predict\\" + df["gender"] + "\\" + df["file"]

    # pixels
    df['pixels'] = df['exact_file'].apply(retrievePixels)

    return df


def features_prediction():
    features = []
    pixels = create_data()['pixels'].values
    for i in range(0, pixels.shape[0]):
        features.append(pixels[i])

    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)

    features = features / 255
    return features

#load model & predict
class MasculinizedFeminized:

    def __init__(self, models):
        self.models = models
        self.pred = create_data().copy()

    def predictions(self):
        for model in self.models:
            #load model
            model_predict = load_model(model+".hdf5")
            #predict
            self.pred[model] = model_predict.predict((features_prediction()))
        self.pred["mean_predictions"] = self.pred.loc[:, self.models].mean(axis=1)
        return self.pred

#Symmetric
def file_to_dic_sym():
    sym = []
    gender = []
    flat_list = []

    path = r"C:\Users\nunok\PycharmProjects\pythonProject2\predict"
    for r, f, d in os.walk(path):
        # folder name
        for folder in f:

            for r, file, d in os.walk(r"C:\Users\nunok\PycharmProjects\pythonProject2\predict\%s" % (folder)):
                # filename in folder
                for img in d:
                    flat_list.append(img)

                    if folder == "male":
                        if (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)) and "sym" in img:
                            gender.append("male")
                            sym.append("yes")
                        elif (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)):
                            gender.append("male")
                            sym.append("no")

                    elif folder == "female":
                        if (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)) and "sym" in img:
                            gender.append("female")
                            sym.append("yes")
                        elif (('.jpg' in img) or ('.jpeg' in img) or ('.png' in img)):
                            gender.append("female")
                            sym.append("no")

    dic = {"gender": gender, "sym": sym,"flat_list":flat_list}
    return dic

def create_data_sym():
    df = pd.DataFrame()
    df["file"] = file_to_dic_sym()["flat_list"]
    df["gender"] = file_to_dic_sym()["gender"]
    df["sym"] = file_to_dic_sym()["sym"]


    df['exact_file'] = "C:\\Users\\nunok\\PycharmProjects\\pythonProject2\\predict\\" + df["gender"] + "\\" + df["file"]

    # pixels
    df['pixels'] = df['exact_file'].apply(retrievePixels)

    return df

def features_prediction_sym():
    features = []
    pixels = create_data_sym()['pixels'].values
    for i in range(0, pixels.shape[0]):
        features.append(pixels[i])

    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)

    features = features / 255
    return features

#load model & predict
class sym:

    def __init__(self, models):
        self.models = models
        self.pred = create_data_sym().copy()

    def predictions(self):
        for model in self.models:
            #load model
            model_predict = load_model(model+".hdf5")
            #predict
            self.pred[model] = model_predict.predict(features_prediction_sym())
        self.pred["mean_predictions"] = self.pred.loc[:, self.models].mean(axis=1)
        return self.pred

#% of congruent predictions with effect direction
def acc(df, column, condition):

    dici = {'male': 0, 'female': 0}
    confusion = {'male': [], 'female': []}
    for key, value in dici.items():
        temp = df[df["gender"] == key]
        if condition == "dim":
            temp = temp[temp["condition"].isin(["non_congruent", "congruent"])]
        ele = 0
        soma = 0

        for index in range(len(temp)):

            # print(index)
            # se for par atualiza
            if index % 2 == 0:

                ele = temp.iloc[index, column]

            # impar compara
            else:
                # print(ele,temp.iloc[index,column])
                if temp.iloc[index, column] > ele and condition == "dim":
                    soma += 1
                    confusion[key].append(1)
                    # print(temp.iloc[index,2])


                elif temp.iloc[index, column] <= ele and condition == "dim":
                    confusion[key].append(0)


                elif temp.iloc[index, column] < ele and condition != "dim":
                    soma += 1
                    confusion[key].append(1)

                elif temp.iloc[index, column] >= ele and condition != "dim":
                    confusion[key].append(1)

        dici[key] = 1 - (soma / len(temp))

    results = [dici, confusion]

    return results

