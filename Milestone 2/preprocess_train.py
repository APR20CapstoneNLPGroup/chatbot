##Imports::
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
import keras

from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import multilabel_confusion_matrix

from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
#stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant
from sklearn.metrics import classification_report, confusion_matrix
import pickle

MAX_SEQUENCE_LENGTH = 80

##Helper functions::
import re
def decontracted(phrase):
  """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""

  # specific
  phrase = re.sub(r"won\'t", "will not", phrase)
  phrase = re.sub(r"can\'t", "can not", phrase)
  phrase = re.sub(r"won\’t", "will not", phrase)
  phrase = re.sub(r"can\’t", "can not", phrase)

  # general
  phrase = re.sub(r"n\'t", " not", phrase)
  phrase = re.sub(r"\'re", " are", phrase)
  phrase = re.sub(r"\'s", " is", phrase)
  phrase = re.sub(r"\'d", " would", phrase)
  phrase = re.sub(r"\'ll", " will", phrase)
  phrase = re.sub(r"\'t", " not", phrase)
  phrase = re.sub(r"\'ve", " have", phrase)
  phrase = re.sub(r"\'m", " am", phrase)

  phrase = re.sub(r"n\’t", " not", phrase)
  phrase = re.sub(r"\’re", " are", phrase)
  phrase = re.sub(r"\’s", " is", phrase)
  phrase = re.sub(r"\’d", " would", phrase)
  phrase = re.sub(r"\’ll", " will", phrase)
  phrase = re.sub(r"\’t", " not", phrase)
  phrase = re.sub(r"\’ve", " have", phrase)
  phrase = re.sub(r"\’m", " am", phrase)

  return phrase

#processed_text = decontracted(processed_text)
#print(processed_text)

def remove_special_character(phrase, remove_number=False):
  """remove_special_character takes text and removes special charcters.
     ref: https://stackoverflow.com/a/18082370/4084039"""

  phrase = re.sub("\S*\d\S*", "", phrase).strip()
  if remove_number:
    phrase = re.sub('[^A-Za-z]+', ' ', phrase)
  else:
    phrase = re.sub('[^A-Za-z0-9]+', ' ', phrase)
  return phrase

# processed_text = remove_special_character(processed_text, True)
# print(processed_text)

def remove_stop_words(text):
    stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', \
                "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', \
                'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", \
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', \
                'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', \
                'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', \
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', \
                'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', \
                'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', \
                'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', \
                'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", \
                'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", \
                'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', \
                "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', \
                "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", \
                'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    return ' '.join(e.lower() for e in text.split() if e.lower() not in stopwords)

# ref: https://gist.github.com/sebleier/554280

# processed_text = ' '.join(e.lower() for e in processed_text.split() if e.lower() not in stopwords)
# print(processed_text)

def lemmatize_text(text_data):
  """lem_text takes text and lemmatize it using WordNetLemmatizer.
     ref: https://stackoverflow.com/a/25535348"""
  lem = WordNetLemmatizer()
  n_text = []
  for word in text_data.split(' '):
    n_word = lem.lemmatize(word, pos='a')
    n_word = lem.lemmatize(n_word, pos='v')
    n_text.append(n_word)

  return ' '.join(n_text)

# processed_text = lem_text(processed_text)
# print(processed_text)

def stem_and_stopwords(text):
    stemmer = nltk.stem.SnowballStemmer('english')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 2 and len(re.sub('\d+', '', word.strip('Xx/'))) > 3) ] 
    tokens = map(str.lower, tokens)
    stems = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
    return stems


def clean_text(text):
    text = decontracted(text)
    text = remove_special_character(text)
    text = remove_stop_words(text)
    text = lemmatize_text(text)
    #text = stem_and_stopwords(text)
    return text


def find_max_list_idx(list):
    list_len = [len(i) for i in list]
    return np.argmax(np.array(list_len))

def desc_to_words(desc):   
    words = RegexpTokenizer('\w+').tokenize(desc)
    words = [re.sub(r'([xx]+)|([XX]+)|(\d+)', '', w).lower() for w in words]
    words = list(filter(lambda a: a != '', words))
    return words



### Key caller methods from API::
def preprocess_data(data):
	'''
	#df : dataframe passed after reading input file as pandas df.
	'''
	print('Shape of input data: {}'.format (data.shape))

	#set the right targets and change its datatype::
	target_column = 'Accident Level'
	target={'I':0, 'II':1, 'III':2, 'IV':3, 'V':4, 'VI':5}
	data['target']=data[target_column].map(target)

	df = data[['target', 'Description']]

	#clean the text::
	df.Description = df.Description.apply(clean_text)

	#convert target to category::
	df['target'].astype('category')

	#Get length of longest line::
	max_idx = find_max_list_idx(df['Description'])
	print('\n Longest line at idx: {}'.format(max_idx))
	print('\n Length of longest line: {}'.format(len(df['Description'][max_idx])))
	print('\n Longest line:: {}'.format(df['Description'][max_idx]))
	#MAX_SEQUENCE_LENGTH = len(df['Description'][max_idx])
	return df


def create_save_w2v_embedding(embedding_file, df):
	desc_lines = list()
	lines = df['Description'].values.tolist()

	for line in lines:   
	    tokens = word_tokenize(line)
	    # convert to lower case
	    tokens = [w.lower() for w in tokens]
	    # remove punctuation from each word    
	    table = str.maketrans('', '', string.punctuation)
	    stripped = [w.translate(table) for w in tokens]
	    # remove remaining tokens that are not alphabetic
	    words = [word for word in stripped if word.isalpha()]
	    # filter out stop words    
	    stop_words = set(stopwords.words('english'))
	    words = [w for w in words if not w in stop_words]
	    desc_lines.append(words)

	EMBEDDING_DIM = 100
	# train word2vec model
	model = gensim.models.Word2Vec(sentences=desc_lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
	# vocab size
	words = list(model.wv.vocab)
	print('Vocabulary size: %d' % len(words))

	# save model in ASCII (word2vec) format
	filename = embedding_file
	print('\n: Saving model file: {}'.format(filename))
	model.wv.save_word2vec_format(filename, binary=False)
	return desc_lines


def train_save_w2_model(df, embedding_file, desc_lines, w2v_model_dir):
	embeddings_index = {}
	f = open(os.path.join('', 'embedding_word2vec.txt'),  encoding = "utf-8")
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	# vectorize the text samples into a 2D integer tensor
	tokenizer_obj = Tokenizer()
	tokenizer_obj.fit_on_texts(desc_lines)
	sequences = tokenizer_obj.texts_to_sequences(desc_lines)

	# pad sequences
	word_index = tokenizer_obj.word_index
	print('\n Found %s unique tokens.' % len(word_index))

	desc_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	X = desc_pad
	print('\n Shape of Desc tensor:', X.shape)

	y = pd.get_dummies(df['target']).values
	print('\n Shape of label tensor:', y.shape)

	#Train-Test split:: 90-10:
	x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 1122, shuffle = True)

	print('##'*20, f'\nNumber of rows in training dataset: {x_train.shape[0]}')
	print(f'Number of columns in training dataset: {x_train.shape[1]}')
	print(f'Number of unique words in training dataset: {len(np.unique(np.hstack(x_train)))}')

	print('##'*20, f'\nNumber of rows in test dataset: {x_val.shape[0]}')
	print(f'Number of columns in test dataset: {x_val.shape[1]}')
	print(f'Number of unique words in test dataset: {len(np.unique(np.hstack(x_val)))}')

	#Oversample using imblearn::
	sm = SMOTENC(random_state=111, categorical_features=[0,1,2,3,4])
	X_res, y_res = sm.fit_resample(X, y)
	print('\n Before Sampling: X_train: {}, y_train: {}'.format(X.shape, y.shape))
	print('\n After Sampling: X: {}, y: {}'.format(X_res.shape, y_res.shape))

	#Train-Test split of oversampled data:: ##TODO:
	X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.20, random_state = 1122)
	print('\n After oversampling: X_train: {}, y_train: {}'.format(X_train.shape,y_train.shape))
	print('\n After oversampling: X_test : {}, y_test : {}'.format(X_test.shape,y_test.shape))

	#Create Embedding vector:
	EMBEDDING_DIM =100
	num_words = len(word_index) + 1
	embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

	for word, i in word_index.items():
	    if i > num_words:
	        continue
	    embedding_vector = embeddings_index.get(word)
	    if embedding_vector is not None:
	        # words not found in embedding index will be all-zeros.
	        embedding_matrix[i] = embedding_vector
	print('\n Embedding Matrix shape: {}'.format(embedding_matrix.shape))

	#Create DL N/w:
	model = Sequential()
	embedding_layer = Embedding(num_words,
	                            EMBEDDING_DIM,
	                            embeddings_initializer=Constant(embedding_matrix),
	                            input_length=MAX_SEQUENCE_LENGTH,
	                            trainable=False)

	model.add(embedding_layer)
	model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
	model.add(SpatialDropout1D(0.2))
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(5, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('\n Model Summary'.format(model.summary()))

	print('\n\n ... Fitting the model...\n')
	h = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val), verbose=2, shuffle=True)


	#Evaluate the model:
	f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 6))
	f.suptitle('Monitoring the performance of the model')
	loss = 'loss'
	val_loss = 'val_loss'
	accuracy = 'accuracy'
	val_accuracy = 'val_accuracy'
	ax1.plot(h.history['loss'], label = 'Train')
	ax1.plot(h.history['val_loss'], label = 'Test')
	ax1.set_title('Model Loss')
	ax1.legend(['Train', 'Test'])
	ax2.plot(h.history[accuracy], label = 'Train')
	ax2.plot(h.history[val_accuracy], label = 'Test')
	ax2.set_title('Model Accuracy')
	ax2.legend(['Train', 'Test'])
	plt.show()

	#Test the model:
	print('\n\n Testing the model...\n\n')
	score, acc = model.evaluate(x_val, y_val, batch_size=10)
	print('\nTest score:', score)
	print('\nTest accuracy:', acc)
	print("\nAccuracy: {0:.2%}".format(acc))


	#CR and CM::
	y_pred = model.predict_proba(X_test, batch_size=128, verbose=1)
	y_pred_bool = np.argmax(y_pred, axis=1)
	y_test_bool = np.argmax(y_test, axis=1)
	print(classification_report(y_test_bool, y_pred_bool))

	confusion_matrix(y_test_bool, y_pred_bool)

	sns.heatmap(confusion_matrix(y_test_bool, y_pred_bool), annot=True)
	report = classification_report(y_test_bool, y_pred_bool, output_dict=True)
	multilabel_confusion_matrix(y_test_bool, y_pred_bool, labels=[0,1, 2, 3, 4]  )	

	labels = [0,1,2,3,4]
	confusion = multilabel_confusion_matrix(y_test_bool, y_pred_bool, labels=labels)

	# Plot confusion matrix 
	fig = plt.figure(figsize = (14, 8))
	for i, (label, matrix) in enumerate(zip(labels, confusion)):
	    plt.subplot(f'23{i+1}')
	    labels = [f'Accident Level:{label}', label]
	    sns.heatmap(matrix, annot = True, square = True, fmt = 'd', cbar = False, cmap = 'Blues', 
	                xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)
	    plt.title(labels[0])

	plt.tight_layout()
	plt.show()

	print('\n Saving w2v mode::')
	#pickle.dump(model, open(w2v_model_file, 'wb')) 
	model.save('w2v_model_dir')


#Predict:
def predict(input_str, w2v_model_dir, w2v_embedding_file):
	loaded_model = keras.models.load_model(w2v_model_dir)
	ip = clean_text(input_str)
	embeddings_index = {}
	f = open(os.path.join('', w2v_embedding_file),  encoding = "utf-8")
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()

	tokenizer_obj = Tokenizer()
	tokenizer_obj.fit_on_texts(ip)
	sequences = tokenizer_obj.texts_to_sequences(ip)
	# pad sequences
	word_index = tokenizer_obj.word_index
	print('\n Found %s unique tokens.' % len(word_index))

	desc_pad = pad_sequences(sequences, maxlen=30)

	ip_padded = desc_pad
	y_pred = loaded_model.predict_proba(ip_padded, batch_size=32, verbose=1)
	y_pred_bool = np.argmax(y_pred, axis=0)
	#print(np.argmax(y_pred_bool))
	prediction = (np.argmax(y_pred_bool))
	print('\nPrediction class: {}'.format(prediction))
	return prediction