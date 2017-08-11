import requests
import base64
import os
import simplejson.scanner

URL = 'http://localhost:5900'


def query_server(img_path):
    with open(img_path, 'rb') as f:
        result = requests.post(URL, json={
            'img': base64.b64encode(f.read()).decode('utf-8')
        })
        try:
            result = result.json()
        except simplejson.scanner.JSONDecodeError:
            print('Cannot decode JSON: ')
            print(result.text)
            exit(1)
        return result['label']


for img in os.listdir('demo_imgs'):
    img = os.path.join('demo_imgs', img)
    print('query "{}" -> {}'.format( img, query_server(img) ))
