import numpy as np 
import nltk
# import keras
# from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
# from keras.optimizers import Adam 
# from keras.models import Model
# from keras.models import Sequential
# from keras.layers import Activation, Dense
# from keras.preprocessing import sequence
# from keras.layers import concatenate

import pickle
with open("vocabulary_movie", "rb") as f:
	vocabulary = pickle.load(f)

weights_file = "my_model_weights.h5"
unknown_token = "something"
maxlen_input = 50
word_embedding_size = 100
sentence_embedding_size = 300
dictionary_size = 7000

new_q = open("dataset/new_q.txt").read().lower()
new_ans = open("dataset/new_ans.txt").read().lower()
# print(new_q)


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


name_of_computer = "Manish"
def preprocess(raw_word, name):
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
      
    return raw_word

new_q = preprocess(new_q, "manish")
# new_ans = preprocess(new_ans, "manish")

a = tokenize(new_q)
print(a[])




# # the neural network
# ad = Adam(lr=0.00005) 

# input_context = Input(shape=(maxlen_input,), dtype='int32', name='the_context_text')
# input_answer = Input(shape=(maxlen_input,), dtype='int32', name='the_answer_text_up_to_the_current_token')
# LSTM_encoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode_context')
# LSTM_decoder = LSTM(sentence_embedding_size, kernel_initializer= 'lecun_uniform', name='Encode_answer_up_to_the_current_token')
# if os.path.isfile(weights_file):
#     Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, input_length=maxlen_input, name='Shared')
# else:
#     Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size, weights=[embedding_matrix], input_length=maxlen_input, name='Shared')
# word_embedding_context = Shared_Embedding(input_context)
# context_embedding = LSTM_encoder(word_embedding_context)

# word_embedding_answer = Shared_Embedding(input_answer)
# answer_embedding = LSTM_decoder(word_embedding_answer)

# merge_layer = concatenate([context_embedding, answer_embedding], axis=1, name='concatenate_the_embeddings_of_the_context_and_the_answer_up_to_current_token')
# out = Dense(int(dictionary_size/2), activation="relu", name='relu_activation')(merge_layer)
# out = Dense(dictionary_size, activation="softmax", name='likelihood_of_the_current_token_using_softmax_activation')(out)

# model = Model(inputs=[input_context, input_answer], outputs = [out])

# model.compile(loss='categorical_crossentropy', optimizer=ad)

# # plot_model(model, to_file='model_graph.png')    

# # if os.path.isfile(weights_file):
# model.load_weights(weights_file)