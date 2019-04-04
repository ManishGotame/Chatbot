from fbchat import Clien
from fbchat.models import *
# from Inference import Facebook_bot
from Jarvis_v2 import Facebook_bot


user_name = "9808002398"
password = "airmon-ng Manishgtm"
# friend_id = "100009311778942" # prasanna
# friend_id = "100008206456265" #` Santosh 


def Chatbot():
  try:
      prob = 0 
      client = Client(user_name, password)

      # mine
      friend_id = client.uid

      # if not client.isLoggedIn():
          # client.login(user_name, password)

      client.send(Message(text="Jarvis is running now."), thread_id=friend_id, thread_type=ThreadType.USER)


      lastMessage = client.fetchThreadMessages(thread_id= friend_id , limit=1, before=None) # the last message
      # this one should be sent by me.
      for last_msg in lastMessage:
          last_msg_data = last_msg.text

      while True:
          new_message = client.fetchThreadMessages(thread_id= friend_id , limit=1, before=None)

          for new_msg in new_message:
              new_msg_data = new_msg.text

          only_for_condition = new_msg_data + " "

          if only_for_condition == last_msg_data:
              pass

          else:
              if prob > 0.02:
                  bot_chat_input = str(response_msg) + " " + str(new_msg_data)
              else:
                  bot_chat_input = new_msg_data

              print()
              print(str(friend_id) + ":" + str(new_msg_data))
  
              if new_msg_data == "<bye>":
                client.send(Message(text="Hasta la vista,baby"), thread_id=friend_id, thread_type=ThreadType.USER)
                break
              '''
                new_msg_data is the input
              '''
              try:
                response_msg, prob = Facebook_bot(bot_chat_input)
                response_msg = "JARVIS: " + str(response_msg)
                print(response_msg)
                client.send(Message(text=response_msg), thread_id=friend_id, thread_type=ThreadType.USER)
  
                last_msg_data = response_msg
              except:
                pass


      client.logout()
  except Exception as e:
      print(e)
      pass

if __name__ == '__main__':
  Chatbot()