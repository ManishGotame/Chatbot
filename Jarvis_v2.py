import pickle 
import re
import numpy as np
import keras
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.preprocessing import sequence
from keras.layers import concatenate
import os
import nltk
import cv2
from datetime import datetime, date
import time
import pyautogui
pyautogui.FAILSAFE = False



weights_file = "my_model_weights.h5"
unknown_token = "something"
maxlen_input = 50
word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000

new_vocab = []

with open("vocabulary_movie", "rb") as f:
	vocabulary = pickle.load(f)
	for x in vocabulary:
		new_vocab.append(x[0])

 # creators method


import pickle
with open("vocabulary_movie", "rb") as f:
	vocabulary = pickle.load(f)

def tokenize(sentences):
	# Tokenizing the sentences into words:
	tokenized_sentences = nltk.word_tokenize(sentences)
	# sentences = re.sub('([.,!?()])', r' \1 ', sentences.lower()))
	# tokenized_sentences = sentences.split()
	index_to_word = [x[0] for x in vocabulary]
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
	tokenized_sentences = [w if w in word_to_index else unknown_token for w in tokenized_sentences]
	X = np.asarray([word_to_index[w] for w in tokenized_sentences])
	s = X.size
	Q = np.zeros((1,maxlen_input))
	if s < (maxlen_input + 1):
		Q[0,- s:] = X
	else:
		Q[0,:] = X[- maxlen_input:]

	return Q


ad = Adam(lr=0.00005) 

input_context = Input(shape=(maxlen_input,), dtype='int32', name='the_context_text')
input_answer = Input(shape=(maxlen_input,), dtype='int32', name='the_answer_text_up_to_the_current_token')
LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode_context')
LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode_answer_up_to_the_current_token')
if os.path.isfile(weights_file):
	Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, name='Shared')
else:
	Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix], input_length=maxlen_input, name='Shared')
word_embedding_context = Shared_Embedding(input_context)
context_embedding = LSTM_encoder(word_embedding_context)

word_embedding_answer = Shared_Embedding(input_answer)
answer_embedding = LSTM_decoder(word_embedding_answer)

merge_layer = concatenate([context_embedding, answer_embedding], axis=1, name='concatenate_the_embeddings_of_the_context_and_the_answer_up_to_current_token')
out = Dense(int(dictionary_size/2), activation="relu", name='relu_activation')(merge_layer)
out = Dense(dictionary_size, activation="softmax", name='likelihood_of_the_current_token_using_softmax_activation')(out)

model = Model(inputs=[input_context, input_answer], outputs = [out])

model.compile(loss='categorical_crossentropy', optimizer=ad)

# plot_model(model, to_file='model_graph.png')    

# if os.path.isfile(weights_file):
model.load_weights(weights_file)

# special functions
def special_access_subroutine():
	from custom_screengrab import GRAB_screen
	# print("Access Granted")
	# print("Command mode")

	img = GRAB_screen(768,1366)
	date_today = date.today()
	time_today = time.time()
	# print(date_today)
	# print(time_today)

	file_name = "file-" + str(date_today) + "-" + str(time_today) + ".jpg"

	cv2.imwrite("sent_images/"+file_name, img)
	return file_name

def camera_routine():
	cap = cv2.VideoCapture(0)

	ret, img = cap.read()

	date_today = date.today()
	time_today = time.time()
	# print(date_today)
	# print(time_today)

	file_name = "file-" + str(date_today) + "-" + str(time_today) + ".jpg"

	cv2.imwrite("sent_images/"+ file_name, img)
	return file_name	

def lockdown_function():
	# pyautogui didn't work
	import ctypes
	ctypes.windll.user32.LockWorkStation()



def command_set(splitted_text):
	for each_word in splitted_text:
		if each_word in ["screenshot"]:
			screenshot_function = True
			file_name = special_access_subroutine()

		if each_word in ["camera"]:
			screenshot_function = True
			file_name = camera_routine()
		if each_word in ['lockdown']:
			lockdown_function()

		if each_word in ['click']:
			pyautogui.press('space')

# ends here


def greedy_decoder(input): # let's leave the algorithm as it is and tmrw we will decipher it
	flag = 0
	prob = 1
	ans_partial = np.zeros((1,maxlen_input))
	ans_partial[0, -1] = 2  #  the index of the symbol BOS (begin of sentence)
	for k in range(maxlen_input - 1):
		ye = model.predict([input, ans_partial])
		yel = ye[0,:]
		p = np.max(yel)
		mp = np.argmax(ye)
		ans_partial[0, 0:-1] = ans_partial[0, 1:]
		ans_partial[0, -1] = mp
		if mp == 3:  #  he index of the symbol EOS (end of sentence)
			flag = 1
		if flag == 0:    
			prob = prob * p

	text = ''
	for k in ans_partial[0]:
		k = k.astype(int)
		if k < (dictionary_size-2):
			w = vocabulary[k]
			text = text + w[0] + ' '
	return(text, prob)


name_of_computer = "Manish"    
# def chat_preprocess(input_text)
def preprocess(raw_word2, name):
	raw_word = ""

	for each_word in raw_word2:
		if ord(each_word) < 128:
			raw_word = raw_word + each_word

	try:
		l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,', 'EOS', 'BOS', 'eos', 'bos']
		l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ', '.', ',', '', '', '', '']
		l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
		l4 = ['jeffrey','fred','benjamin','paula','walter','rachel','andy','helen','harrington','kathy','ronnie','carl','annie','cole','ike','milo','cole','rick','johnny','loretta','cornelius','claire','romeo','casey','johnson','rudy','stanzi','cosgrove','wolfi','kevin','paulie','cindy','paulie','enzo','mikey','i\97','davis','jeffrey','norman','johnson','dolores','tom','brian','bruce','john','laurie','stella','dignan','elaine','jack','christ','george','frank','mary','amon','david','tom','joe','paul','sam','charlie','bob','marry','walter','james','jimmy','michael','rose','jim','peter','nick','eddie','johnny','jake','ted','mike','billy','louis','ed','jerry','alex','charles','tommy','bobby','betty','sid','dave','jeffrey','jeff','marty','richard','otis','gale','fred','bill','jones','smith','mickey']    

		raw_word = raw_word.lower()
		raw_word = raw_word.replace(', ' + name_of_computer, '')
		raw_word = raw_word.replace(name_of_computer + ' ,', '')

		for j, term in enumerate(l1):
			raw_word = raw_word.replace(term,l2[j])
			
		for term in l3:
			raw_word = raw_word.replace(term,' ')
		
		for term in l4:
			raw_word = raw_word.replace(', ' + term, ', ' + name)
			raw_word = raw_word.replace(' ' + term + ' ,' ,' ' + name + ' ,')
			raw_word = raw_word.replace('i am ' + term, 'i am ' + name_of_computer)
			raw_word = raw_word.replace('my name is' + term, 'my name is ' + name_of_computer)
		
		for j in range(30):
			raw_word = raw_word.replace('. .', '')
			raw_word = raw_word.replace('.  .', '')
			raw_word = raw_word.replace('..', '')
		   
		for j in range(5):
			raw_word = raw_word.replace('  ', ' ')
			
		if raw_word[-1] !=  '!' and raw_word[-1] != '?' and raw_word[-1] != '.' and raw_word[-2:] !=  '! ' and raw_word[-2:] != '? ' and raw_word[-2:] != '. ':
			raw_word = raw_word + ' .'
		
		if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
			raw_word = 'what ?'
		
		if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
			raw_word = 'i do not want to talk about it .'
	except:
		pass
		raw_word = "unk"
		  
	return raw_word


def Interative_mode():
	prob = 0 

	access = False

	while True:
		access_count = 0
		new_chat_input = input("input: ")
		new_chat_input = new_chat_input.lower()

		if new_chat_input == "<bye>":
			break

		if prob > 0.02:
			new_chat_input = str(response_text) + ' ' + str(new_chat_input)
		else:
			pass

		# response_text = chat_preprocess(new_chat_input)

		new_chat_input = preprocess(new_chat_input, "Jarvis")
		print(new_chat_input)

		# The special access
		'''
			for now, We're not using any AI but a simple method to find out the access user. Please update it in the future
		'''

		# print(new_chat_input.split())
		splitted_text = new_chat_input.split()

		for each_word in splitted_text:
			if each_word in ["i","will","take","the","access","from","here"]:
				access_count = access_count + 1
			elif each_word in ["back","to","normal","jarvis"]:
				access_count = access_count + 1
			elif each_word in ["override","02000x"]:
				access_count = access_count + 1

		if access_count == 7 or access_count == 2:
			# print("let's do it")
			access = True
		elif access_count == 4:
			access = False

		# Ends here maybe

		# if access == True:
		if access == True:
			command_set(splitted_text)
		# else:

		new_chat_input = tokenize(new_chat_input)

		predout, prob = greedy_decoder(new_chat_input)
		start_index = predout.find('EOS')

		response_text = predout[4:(start_index-1)]

		print(response_text, prob)

		# chat_mode(access,new_chat_input)


def Facebook_bot(chat_input,access):
	file_name = "sth"
	screenshot_function = False
	# access = False
	access_count = 0
	chat_input = chat_input.lower()

	chat_input = preprocess(chat_input, "Manish")

	print(access)

	# The special access

	'''
		for now, We're not using any AI but a simple method to find out the access user. Please update it in the future
	'''

	splitted_text = chat_input.split()

	if access == True:
		for each_word in splitted_text:
			if each_word in ["screenshot"]:
				screenshot_function = True
				file_name = special_access_subroutine()

			if each_word in ["camera"]:
				screenshot_function = True
				file_name = camera_routine()
			if each_word in ['lockdown']:
				lockdown_function()

	# Ends here maybe

	new_chat_input = tokenize(chat_input)


	predout, prob = greedy_decoder(new_chat_input)
	start_index = predout.find('EOS')
	response_text = predout[4:(start_index-1)]
	
	if response_text == "fred":
		response_text = "Jarvis ."

	if screenshot_function == False:
		file_name = None
	return response_text, prob, file_name

# Interative_mode()
