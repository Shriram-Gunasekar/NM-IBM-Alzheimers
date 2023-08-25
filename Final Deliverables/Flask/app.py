import numpy as np
import PIL
import os
from keras.utils import load_img, img_to_array
from keras.preprocessing import image
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
#Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
# from tensorflow.keras.layers import BatchNormalization

#global graph
#tf.compat.v1.enable_eager_execution
#tf.compat.v1.disable_eager_execution()
#sess = tf.compat.v1.Session()

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#graph = tf.compat.v1.get_default_graph()

app = Flask(__name__)
#set_session(sess)
#load your trained model
model = keras.models.load_model('adp.h5')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('alzheimers.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image'] 
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = load_img(file_path, target_size=(180,180,3))
        x = img_to_array(img)
        x = np.expand_dims(x,axis=0)
        # image_data = preprocess_input(x)
        #with graph.as_default():
        #    set_session(sess)
        preds = model.predict(preprocess_input(x))
        preds = np.argmax(preds, axis=1)
        print(preds)
        index = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
        text = str(index[preds[0]])
        print(text)
        return render_template('predict.html', text=text)
    return render_template("predict.html",text='Please upload an image first')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)