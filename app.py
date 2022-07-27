from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import base64
import io
import NSTmodel


app = Flask(__name__)

@app.route('/')
def api():
    return 'api running'

@app.route('/get_transfer_image', methods = ['POST'])
def getTransferImage():
    if request.method == 'POST':
        model = NSTmodel.NST()
        # content image
        fileStrContent = request.files['contentImage'].read()
        npContentImage = np.fromstring(fileStrContent, np.uint8)
        contentImage = cv2.imdecode(npContentImage, cv2.IMREAD_UNCHANGED)
        contentImage = cv2.cvtColor(contentImage , cv2.COLOR_BGR2RGB)
        model.loadContentImage(contentImage)
        #style image
        fileStrStyle = request.files['styleImage'].read()
        npStyleImage = np.fromstring(fileStrStyle, np.uint8)
        styleImage = cv2.imdecode(npStyleImage, cv2.IMREAD_UNCHANGED)
        styleImage = cv2.cvtColor(styleImage , cv2.COLOR_BGR2RGB)
        model.loadStyleImage(styleImage)
        #setting images
        model.setLoadedImages()  
        model.Extractor()
        #main processing
        img = model.masterCall()
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img = base64.b64encode(rawBytes.read())
        response = jsonify({'status':str(img)})
        response.headers.add("Access-Control-Allow-Origin", "*")
    return response
    
    
    
    #  filestr = request.files['file'].read()
    
    #  ImageContent 
    # #  npimg = np.fromstring(filestr, np.uint8)
    # #  img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    # #  img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    # #  img = Image.fromarray(img.astype("uint8"))
    
    
    
    
    
    # if request.method == 'POST':
    #  model = NSTmodel.NST()
    #  model.loadContentImage(session['content_image'])
    #  model.loadStyleImage(session['style_image'])
    #  model.setLoadedImages()
    #  model.Extractor()
    #  img = model.masterCall()
    #  rawBytes = io.BytesIO()
    #  img.save(rawBytes, "JPEG")
    #  rawBytes.seek(0)
    #  img = base64.b64encode(rawBytes.read())

    # return jsonify({'status':str(img)})




        
     
 
  

if __name__ == "__main__":
    app.run()