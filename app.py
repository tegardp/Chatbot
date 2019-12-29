from bot.bot import Chatbot
import json

with open('bot/data/intents.json') as file:
    data = json.load(file)


bot = Chatbot()
bot.set_training(data)

def chat():
    print('ketik "quit" untuk berhenti')
    while True:
        inp = input(">>> ")
        if inp.lower() == "quit":
            break

        print(bot.get_response(inp))

chat()