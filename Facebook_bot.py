from fbchat import Client 
from fbchat.models import *
# from Inference import Facebook_bot
from Jarvis_v2 import Facebook_bot


user_name = "manish.gotame@yahoo.com"
password = "airmon-ng interface"

# user_name = "manishgt194@gmail.com"
# password = "airmon-ng Manishgtm"
# friend_id = "100009311778942" # prasanna
# friend_id = "100008206456265" # Santosh 
# friend_id = "100009251786386" # present
friend_id = "100010850250687" # Manish
# friend_id = "100009418603688" # roshan 

prob = 0
access = True

try:
	client = Client(user_name, password)

	# mine
	# friend_id = client.uid

	# ends here

	# if not client.isLoggedIn():
		# client.login(user_name, password)

	# client.send(Message(text="Jarvis is running now."), thread_id=friend_id, thread_type=ThreadType.USER)


	lastMessage = client.fetchThreadMessages(thread_id= friend_id , limit=1, before=None) # the last message
	# this one should be sent by me.
	for last_msg in lastMessage:
		last_msg_data = last_msg.text


	while True:
		new_message = client.fetchThreadMessages(thread_id= friend_id , limit=1, before=None)

		for new_msg in new_message:
			new_msg_data = new_msg.text

		only_for_condition = str(new_msg_data)
		# print("last message:",len(last_msg_data))
		# print("last message:", last_msg_data)
		# print("new message", len(new_msg_data))
		# print("new_message:", new_msg_data)


		if only_for_condition == last_msg_data:
			pass

		else:
			# print("responded")
			if prob > 0.002:
				bot_chat_input = str(response_msg) + " " + str(new_msg_data)
			else:
				bot_chat_input = new_msg_data
		
			print()
			print(str(friend_id) + ":" + str(new_msg_data))

			if new_msg_data == "<bye>":
				client.send(Message(text="Jarvis out."), thread_id=friend_id, thread_type=ThreadType.USER)
				break
			'''
				new_msg_data is the input
			'''
			try:
				client.setTypingStatus(TypingStatus.TYPING, thread_id=friend_id, thread_type=ThreadType.USER)
				
				'''
				try:
					response_msg, prob = Facebook_bot(bot_chat_input)
					client.send(Message(text=response_msg), thread_id=friend_id, thread_type=ThreadType.USER)
				except:
					response_msg, prob, file_name = Facebook_bot(bot_chat_input)
					client.sendLocalImage(('sent_images/'+ file_name), message=Message(text=response_msg), thread_id=friend_id, thread_type=ThreadType.User)
				'''


				response_msg, prob, file_name = Facebook_bot(bot_chat_input,access)

				if file_name != None:
					print("this should be here")
					client.sendLocalImage(('sent_images/'+ file_name), message=Message(text="You asked for this?"), thread_id=friend_id, thread_type=ThreadType.USER)
				else:
					client.send(Message(text=response_msg), thread_id=friend_id, thread_type=ThreadType.USER)


				
				print("Jarvis: " + str(response_msg))

				last_msg_data = response_msg
			except Exception as e:
				print(e)
				pass


	client.logout()
except Exception as e:
	print("error",e)
	pass
