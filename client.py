import os
import base64
import requests
import json

url = 'http://localhost:5000/predict'
imgPath = os.getcwd() + '/49bb8caf-4c31-4bae-b345-1b03f5fd8e34.JPG'
with open(imgPath, "rb") as imageFile:
    str = base64.b64encode(imageFile.read()).decode('utf-8')

    data = {}
    data['img'] = str
    r = requests.post(url,json=json.dumps(data))