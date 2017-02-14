#!/usr/local/bin/python
from PIL import Image
import ImageDraw, ImageFont
import numpy as np

def demo():
	text='A'
	size=32
	color='white'
	fontPath = '/data/fonts/DroidSans.ttf' #/FreeSansBold.ttf'
	font = ImageFont.truetype(fontPath, size)
	size2 = font.getsize(text)
	padding = 2
	padding_y = 3
	size2 = (size2[0],size2[1]+padding_y) # add margin
	size2 = (size + padding, size + padding) # ignore rendered font size!
	img = Image.new('L', size2)# # grey
	# img = Image.new('RGBA', size2, (0, 0, 0, 0))
	draw = ImageDraw.Draw(img)
	draw.text((padding, -padding), text, font=font, fill=color)
	x=np.array(img)
	img.show()

demo()
print("Just a demo; The ocr data is created on the fly in letter.py when training")
