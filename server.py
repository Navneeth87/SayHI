import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
from predict import final
app = Flask(__name__)
CORS(app)
model = pickle.load(open('gender.pkl','rb'))
@app.route("/", methods=["GET"])
def main():
    return "Hello there"

@app.route('/api',methods=['POST'])
def predict():
   data = request.data
   f = open('./file.wav', 'wb')
   f.write(data)
   f.close()
   feat = final()
   print("feat rcvd")
   print(model.predict(feat))
   return "Hello there"
if __name__ == '__main__':
    app.run(port=5000, debug=True)