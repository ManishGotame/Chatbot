#imports 
import numpy as np
import os
# np.set_printoptions(threshold=np.nan) # to see the entire numpy data for debugging 
import re
import queue
Queue = queue
import keras
from keras.models import Model
from keras.layers import Dense, Input, Embedding, LSTM, Flatten, Masking

from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from session_3 import encoder_tokens, decoder_tokens, enc_chars_to_int, enc_int_to_chars, dec_int_to_chars, dec_chars_to_int
from session_3 import input_max_length, encoder_input_data, decoder_input_data

# def inference_prediction(input_seq):
embedding_size = 100

#encoder model 
encoder_inputs = Input(shape=(None,))
en_x=  Embedding(encoder_tokens, embedding_size, mask_zero=True)(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

#Decoder model
decoder_inputs = Input(shape=(None,))
dex=  Embedding(decoder_tokens, embedding_size, mask_zero=True)
final_dex= dex(decoder_inputs)
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)
decoder_dense = Dense(decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['acc']) # keep an eye out for this one

# model.summary()



'''
	model-0220 was comparatively better 

'''
# json_file = open('seq2seq_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()

# loaded_model = model_from_json(loaded_model_json)
# loaded_model = model_0()
# load weights into new model
# model.load_weights("weights/sgd2_new_cp-em-0225.ckpt")
# model.load_weights("weights/new_cp-em-0960.ckpt")
# model.load_weights("weights/sgd2_new_cp-em-0225.ckpt")
# model.load_weights("weights/final2_sgd2_new_cp-em-0145.ckpt") # this one is interesting to talk with
model.load_weights("weights/zjarvis-sgd2_new_cp-em-0070.ckpt")





# inference setup 
def inference_prediction(input_seq):
	encoder_model = Model(encoder_inputs, encoder_states)

	decoder_state_input_h = Input(shape=(50,))
	decoder_state_input_c = Input(shape=(50,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

	final_dex2 = dex(decoder_inputs)

	decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
	decoder_states2 = [state_h2, state_c2]
	decoder_outputs2 = decoder_dense(decoder_outputs2)

	decoder_model = Model(
			[decoder_inputs] + decoder_states_inputs,
			[decoder_outputs2] + decoder_states2
		)

	# the decoder sequence

	states_value = encoder_model.predict(input_seq)

	target_seq = np.zeros((1,1))

	target_seq[0,0] = dec_chars_to_int["<start>"]

	stop_condition = False
	decoded_sentence = ''

	while not stop_condition:
		output_tokens, h, c = decoder_model.predict(
				[target_seq] + states_value
			)

		# print(output_tokens)
		np.save("Obsolete/pred.npy", output_tokens)

		sampled_token_index = np.argmax(output_tokens[0, -1, :]) # a different predict_classes method
		sampled_word = dec_int_to_chars[sampled_token_index]

		if sampled_word == "<pad>" or sampled_word == "<end>":
			pass
		else:
			decoded_sentence += ' ' + sampled_word
		# print(decoded_sentence)
		# break

		if (sampled_word == "<end>" or len(decoded_sentence) > input_max_length):
			stop_condition = True

		target_seq = np.zeros((1,1))
		target_seq[0, 0] = sampled_token_index

		states_value = [h, c]

	return decoded_sentence



def inference():
	for seq_index in [234,4544, 4444, 8998, 12323, 15897, 3434, 9783]:
		input_seq = encoder_input_data[seq_index: seq_index+1] # 0 to 1 = 0, 1 to 2 = 1  # without seq_index+1, it creates an error
		# print(input_seq.shape)
		expected_output = decoder_input_data[seq_index: seq_index+1]

		reversed_input = []

		for each in input_seq:
			reversed_input_data = each[::-1]
			for each in reversed_input_data:
				if each == 0:
					pass
				else:
					reversed_input.append(enc_int_to_chars[each])

		input_text = " ".join(reversed_input)


		expected_output_a = []
		for each in expected_output:
			for each_s in each:
				if each_s == 0:
					pass
				else:
					expected_output_a.append(dec_int_to_chars[each_s])

		expected_text = " ".join(expected_output_a)


		# print(reversed_input)
		# input_text = " ".join(a)


		# reverse the sentence for input
		print()
		print("input sentence: ",input_text)
		print("--")
		decoded_sentence = inference_prediction(input_seq)
		print("Expected Sentence: ", expected_text)
		print("--")
		print("output sentence: ",decoded_sentence)


def Chat_interface():
	# from inference import decoder_sequence
	print("Chat Session (Press CTRL + C to end) ")
	print()


	while True:
		chat_input = input("Input: ")

		chat_input = re.sub('([.,!?()])', r' \1 ', chat_input.lower())
		chat_input = chat_input.split()

		chat_input = chat_input[::-1] # reverse the sentence
		chat_input = chat_padding(chat_input, input_max_length) # padding

		# chat_input = int_data(chat_input, enc_chars_to_int)

		chat_input_array = []
		for each in chat_input:
			try:
				chat_input_array.append(enc_chars_to_int[each])
			except:
				chat_input_array.append(enc_chars_to_int["<unk>"])

		chat_input_array = np.array([chat_input_array])
		# print(chat_input_array.shape)
		# print(chat_input_array)

		decoded_sentence = inference_prediction(chat_input_array)
		print("Output: ",decoded_sentence)
		print()


def Facebook_bot(chat_input):
	def chat_padding(unpadded_sentence, max_length):
		# padded_sentences = []
		second = [] # I ran out of variable names!
		for i in range(max_length):
			try:
				second.append(unpadded_sentence[i])
			except:
				second.append("<pad>")
		return second
	chat_input = re.sub('([.,!?()])', r' \1 ', chat_input.lower())
	chat_input = chat_input.split()
	chat_input = chat_input[::-1] # reverse the sentence

	chat_input = chat_padding(chat_input, input_max_length) # padding
	
	chat_input_array = []
	for each in chat_input:
		try:
			chat_input_array.append(enc_chars_to_int[each])
		except:
			chat_input_array.append(enc_chars_to_int["<unk>"])
	
	chat_input_array = np.array([chat_input_array])
	# print(chat_input_array.shape)
	# print(chat_input_array)
	decoded_sentence = inference_prediction(chat_input_array)
	
	return decoded_sentence


# Chat_interface()
inference()
# response = Facebook_bot("do you want me to shut you down")
# print(response)