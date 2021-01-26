#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import requests
from flask import Flask, request

import spacy
spacy.load("en_core_web_sm")
app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():

    # cuando el endpoint este registrado como webhook, debe mandar de vuelta
    # el valor de 'hub.challenge' que recibe en los argumentos de la llamada

    if request.args.get('hub.mode') == 'subscribe' \
        and request.args.get('hub.challenge'):
        if not request.args.get('hub.verify_token') \
            == os.environ['VERIFY_TOKEN']:
            return ('Verification token mismatch', 403)
        return (request.args['hub.challenge'], 200)

    return ('Hello world', 200)


@app.route('/', methods=['POST'])
def webhook():

    # endpoint para procesar los mensajes que llegan

    data = request.get_json()

    # log(data)  # logging, no necesario en produccion

    inteligente = True

    if data['object'] == 'page':

        for entry in data['entry']:
            for messaging_event in entry['messaging']:

                if messaging_event.get('message'):  # alguien envia un mensaje

                    sender_id = messaging_event['sender']['id']  # el facebook ID de la persona enviando el mensaje
                    recipient_id = messaging_event['recipient']['id']  # el facebook ID de la pagina que recibe (tu pagina)
                    message_text = messaging_event['message']['text']  # el texto del mensaje

                    if inteligente:
                        chatbot = ChatBot('Chalo')
                        trainer = ChatterBotCorpusTrainer(chatbot)

                        # Train the chatbot based on the spanish corpus

                        trainer.train('chatterbot.corpus.english')

                        response = chatbot.get_response(message_text)

                        send_message(sender_id, response.text)
                    else:
                        send_message(sender_id, 'Hola')

                if messaging_event.get('delivery'):  # confirmacion de delivery
                    pass

                if messaging_event.get('optin'):  # confirmacion de optin
                    pass

                if messaging_event.get('postback'):  # evento cuando usuario hace click en botones
                    pass

    return ('ok', 200)


def send_message(recipient_id, message_text):

    # log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {'access_token': os.environ['PAGE_ACCESS_TOKEN']}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({'recipient': {'id': recipient_id},
                      'message': {'text': message_text}})

    r = requests.post('https://graph.facebook.com/v2.6/me/messages',
                      params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


def log(message):  # funcion de logging para heroku
    print(str(message))
    sys.stdout.flush()


if __name__ == '__main__':
    app.run(debug=True)
