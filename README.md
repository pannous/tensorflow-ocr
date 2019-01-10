# tensorflow-ocr
🖺 OCR using tensorflow with attention, batteries included

# Installation
```
git clone --recursive http://github.com/pannous/tensorflow-ocr
# sudo apt install python3-pip
cd tensorflow-ocr
pip install -r requirements.txt
```

# Evaluation

You can detect the text under your mouse pointer with 
`mouse_prediction.py`

it takes 10 seconds to load the network and startup, then it should return multiple results per second
. 

`text_recognizer.py`

To combine our approach with real world images we forked the [EAST](https://github.com/quasiris/EAST) boundary boxing.

# Customized training

To get started with a minimal example similar to the famous MNIST try
`./train_letters.py` ;
It automatically generates letters for all different font types from your computer in all different shapes and trains on it.

For the full model used in the demo start `./train.py`
