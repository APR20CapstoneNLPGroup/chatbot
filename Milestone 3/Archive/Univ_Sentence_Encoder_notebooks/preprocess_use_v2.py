    
#t = ['AccLevel', 'Pot_AccLevel', 'Cri_Risk']
#target = input("Please select target 'AccLevel', 'Pot_AccLevel', 'Cri_Risk': ")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
import random

import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.utils import resample

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Conv2D, Input, LSTM,Activation, Dropout,Embedding,  MaxPooling1D, Bidirectional
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K

#from tensorflow.keras.layers import Flatten, Dense,  Dense, Activation, Dropout,Embedding,  MaxPooling1D, Conv1D, Bidirectional

#from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

np.random.seed(10)

# NLP Import packages
import nltk; nltk.download('wordnet'); nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string

from string import punctuation
from wordcloud import STOPWORDS


def load_dataset(filename):
  data = pd.read_csv(filename)
  return data
  
 
def data_cleansing_info(data):
  data.drop("Unnamed: 0", axis=1, inplace=True)
  data.rename(columns={'Data':'Date', 'Countries':'Country', 'Accident Level' : 'AccLevel' ,  'Genre':'Gender', 'Employee or Third Party':'Employee type' , 'Potential Accident Level':'Pot_AccLevel', 'Critical Risk':'Cri_Risk'}, inplace=True)
  return data.head(5), data.isnull().sum()

#TEXT CLEANING
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"ur", " your ", text)
    text = re.sub(r" nd "," and ",text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" tkts "," tickets ",text)
    text = re.sub(r" c "," can ",text)
    text = re.sub(r" e g ", " eg ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
    text = re.sub(r" u "," you ",text)
    text = text.lower()  # set in lowercase 

    stop_words =  set(nltk.corpus.stopwords.words('english'))

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return(text)


### Key caller methods from API::
def preprocess_data(data):

    data.drop("Unnamed: 0", axis=1, inplace=True)
    data.rename(columns={'Data':'Date', 'Countries':'Country', 'Accident Level' : 'AccLevel' ,  'Genre':'Gender', 'Employee or Third Party':'Employee type' , 'Potential Accident Level':'Pot_AccLevel', 'Critical Risk':'Cri_Risk'}, inplace=True)
    print(data.head(2))
    data["clean_Description"] = data["Description"].apply(text_cleaning)
    data.drop('Description', axis=1, inplace=True)
    data.rename(columns={'clean_Description':'Description'}, inplace=True)
    print(data.head(2))
    target = 'AccLevel'
    

    df = pd.DataFrame(data, columns=[target, 'Description'])
    print(df.head(2))
    df[target] = df[target].astype('category')
    categories = df[target].cat.categories.tolist()
    pickle.dump( df, open( "processed_data.p", "wb" ))
    print("File saved")
    return df

def train_fit_model(df, embed, embed_size):
    
    df = pickle.load( open( "processed_data.p", "rb" ))
    
    #Train-Test Split
    
    # split the data into train and test set
    trainval, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    # Split train into train-val
    df_train, val = train_test_split(trainval, test_size=0.1, random_state=21, shuffle=True)

    categories = df_train[target].cat.categories.tolist()

    if(target == 'AccLevel'):
        df_train_1 = df_train[df_train[target] == "I"]
        df_train_2 = df_train[df_train[target] == "II"]
        df_train_3 = df_train[df_train[target] == "III"]
        df_train_4 = df_train[df_train[target] == "IV"]
        df_train_5 = df_train[df_train[target] == "V"]

        df_train_2_upsampled = resample(df_train_2, replace = True, n_samples = 223 , random_state = 123)
        df_train_3_upsampled = resample(df_train_3, replace = True, n_samples = 223 , random_state = 123)
        df_train_4_upsampled = resample(df_train_4, replace = True, n_samples = 223 , random_state = 123)
        df_train_5_upsampled = resample(df_train_5, replace = True, n_samples = 223 , random_state = 123)

        df_train_upsampled = pd.concat([df_train_1, df_train_2_upsampled,df_train_3_upsampled, df_train_4_upsampled, df_train_5_upsampled ])
        print(df_train_upsampled[target].value_counts())

        df_train_upsampled_bkup = df_train_upsampled.copy()
        df_train_upsampled_bkup[target].value_counts()

        df_train = df_train_upsampled.copy()

    
    print(df_train_upsampled[target].value_counts())

    df_train_upsampled_bkup = df_train_upsampled.copy()
    df_train_upsampled_bkup[target].value_counts()

    #df_train = df_train_upsampled.copy()
    #pickle.dump( df_train, open( "df_train.p", "wb" ))
    
    # Get the Category Counts
    category_counts = len(df_train[target].cat.categories)
    
    #Converting train dataset to correct format

    train_text = df_train['Description'].tolist()
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]

    train_label = np.asarray(pd.get_dummies(df_train[target]), dtype = np.int8)
    train_label[8]

    #Converting test dataset to correct format

    test_text = df_test['Description'].tolist()
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = np.asarray(pd.get_dummies(df_test[target]), dtype = np.int8)

    #Converting validation dataset to correct format

    val_text = val["Description"].tolist()
    val_text = np.array(val_text, dtype=object)[:, np.newaxis]
    val_label = np.asarray(pd.get_dummies(val[target]), dtype = np.int8)

    
    #Model Building

    #from keras.layers import Dense, Embedding, LSTM, Dropout, MaxPooling1D, Conv1D, Bidirectional

    #input_text = layers.Input(shape=(1,), dtype=tf.string)
    input_text = layers.Input(shape=(1,), dtype="string") #https://github.com/tensorflow/tensorflow/issues/19303

    print(input_text.shape)
    embedding = layers.Lambda(UniversalEmbedding,output_shape=(embed_size,))(input_text)
    print(embedding.shape)
    reshape = layers.Reshape(target_shape=(1, 512 ))(embedding)
    print(reshape.shape)
    drop = (Dropout(0.25))(reshape)
    cnvl = (Conv1D(256, 5, padding = 'same', activation = 'relu', strides = 1))(drop)
    lstm=LSTM(units=128,return_sequences=False)(cnvl)
    #lstm = (Bidirectional(LSTM(units=100, return_sequences=False, recurrent_dropout=0.1)))(drop)
    #attn = SeqSelfAttention(attention_activation='softmax')(lstm)
    #attn = (Activation("softmax"))(lstm)
    dense = layers.Dense(256, activation='relu')(lstm)
    print(dense.shape)

    pred = layers.Dense(category_counts, activation='softmax')(dense)
    model = Model(inputs=[input_text], outputs=pred)

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    #print model summary
    print(model.summary())

    print(type(model))

    tf.keras.utils.plot_model(model, show_shapes = True)

    #Model Training

    with tf.Session() as session:
      K.set_session(session)
      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())
      # Adding callbacks
      es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 0)  
      mc = ModelCheckpoint('model_lstm.h5', monitor = 'val_loss', mode = 'min', save_best_only = True, verbose = 1)
      #logdir = 'log'; tb = TensorBoard(logdir, histogram_freq = 1)
      lr_r = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 0)
      
      cb = [es, lr_r] #https://stackoverflow.com/questions/58030543/list-of-keras-callbacks-generates-error-tuple-object-has-no-attribute-set-mo
      
      h = model.fit(train_text, train_label, validation_data=(val_text, val_label), epochs=30, batch_size=32, callbacks=[es, lr_r, mc])
      
    # #Plotting the Training & Test Accuracy and Loss

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 7.2))
    # f.suptitle('Training & Testing Loss')

    # ax1.plot(h.history['loss'], 'go-' , label = 'Train')
    # ax1.plot(h.history['val_loss'],  'ro-' , label = 'Test')
    # ax1.set_title('Model Loss')
    # ax1.legend(['Train', 'Test'])
    # ax1.set_xlabel("Epochs")
    # ax1.set_ylabel("Loss")

    # ax2.plot(h.history['categorical_accuracy'], 'go-', label = 'Train')
    # ax2.plot(h.history['val_categorical_accuracy'], 'ro-' , label = 'Test')
    # ax2.set_title('Model Accuracy')
    # ax2.legend(['Train', 'Test'])
    # ax2.set_xlabel("Epochs")
    # ax2.set_ylabel("Accuracy")

    # plt.show()



    #Print the Classification Report and Overall Accuracy

    # Evaluate the model
    with tf.Session() as session:
      K.set_session(session)
      session.run(tf.global_variables_initializer())
      session.run(tf.tables_initializer())
      loss, categorical_accuracy = model.evaluate(test_text, test_label, verbose = 0)
      print('Overall Accuracy: {}'.format(categorical_accuracy * 100))
      y_pred = model.predict(test_text, batch_size=10, verbose=1)
      y_pred_bool = np.argmax(y_pred, axis=1)
      y_test_bool = np.argmax(test_label, axis=1)
      print(classification_report(y_test_bool, y_pred_bool))


    model_score = metrics.accuracy_score(y_test_bool, y_pred_bool)
    print(model_score * 100)


    #Print Confusion Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

    cm = confusion_matrix(y_test_bool, y_pred_bool)
    #cm = multilabel_confusion_matrix(y_test_bool, y_pred_bool)

    #Plot the  Confusion Matrix

    # plt.figure(figsize = (10,7))
    # sns.heatmap(cm, annot=True)

    print('\n Trained Model saved ...')
	  
