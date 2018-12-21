#!/usr/bin/env python

import requests
import json
from PIL import Image # Image.open(test_image)
# import cv2 # cv2.imwrite(output_path, img)
# import skimage #skimage.io.imread

class Box(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

image_file='test_image.png'
test_file='test_out.png'
image = Image.open(image_file)

with open(image_file, 'rb') as f:
    r = requests.post('http://0.0.0.0:8769/?json=1', files={'image': f})
    print(r.text)
    ocr=json.loads(r.text)
    for line in ocr['text_lines']:
      print(line)
      b=Box(**line)
      print(b.x0)
      word=image.crop((b.x0-5, b.y0-5, b.x2+15, b.y2+15))
      word.save(open(test_file, 'wb'))

# h = httplib.HTTPConnection('http://0.0.0.0:8769')
# headers={}
# data=[]
# h.request('POST', '/?json=1', data, headers)
# r = h.getresponse()
# result=r.read()  