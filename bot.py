from flask import Flask, request, jsonify

app = Flask(__name__)

import requests

BOT_ID = 's645137b2-d6fa-48bd-a658-c716fa2ab8c0'
API_KEY = 'ee3b55e08d661bd5cd962af531aa764f'

def get_botstar_response(input_text):
    url = f'https://api.botstar.com/v1/me/bots/{BOT_ID}/modules/message'
    print(url)
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'text': input_text
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    return response.json()

# @app.route('/', methods=['POST'])
# def webhook():
#     data = request.json
#     input_text = data['message']['text']
#     botstar_response = get_botstar_response(input_text)
#     print(botstar_response)
#     response = {
#         "message": {
#             "text": botstar_response['message']['text']
#         }
#     }
#     return jsonify(response)
for i in range(10):
    i=input("Enter your message")
    botstar_response = get_botstar_response(i)
    print(botstar_response)

if __name__ == '__main__':
    app.run(debug=True)
