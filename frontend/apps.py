
from flask import Flask,flash, render_template, Response, request , redirect, url_for , session
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import plotly
import pathlib



# learn = load_learner('resnet.pkl')


#     \learn = load_model('model_vgg_own.pkl')

# def classify_image(img_path):
#     # Map predictions to labels
#     mapping = {0: 'Actinic keratosis',
#                1: 'Basal cell carcinoma',
#                2: 'Benign keratosis',
#                3: 'Dermatofibroma',
#                4: 'Melanocytic nevus',
#                5: 'Melanoma',
#                6: 'Squamous cell carcinoma',
#                7: 'Vascular lesion',
#                8: 'Normal'}

#     # Load the image and predict
#     img = load_image.open(img_path)
#     pred_idx,_,probs = learn.predict(img)
#     pred_class = mapping[int(pred_idx)]
#     prob = probs.max()
#     prob = prob.numpy()
#     return pred_class,prob


case = pd.read_csv("../Dataset/Monkey_Pox_Cases_Worldwide.csv")
case_timeline =pd.read_csv("../Dataset/Worldwide_Case_Detection_Timeline.csv")
case_country = pd.read_csv("../Dataset/Daily_Country_Wise_Confirmed_Cases.csv")



#--- BTN UPLOAD ----
UPLOAD_FOLDER = './static/Data'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__, template_folder = "templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_model():
    global model
    model = load_model('model.h5')
    print("Model loaded")

def load_image(img_path):

    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)                   
    img_tensor = np.expand_dims(img_tensor, axis=0)       
    img_tensor /= 255.                                     

    return img_tensor

def prediction(img_path):
    new_image = load_image(img_path)
    
    pred = model.predict(new_image)
    
    print(pred)
    if pred<0.5:
        return "It might be Monkeypox. You should visit a specialist immediately. Thank you."
    else:
        return "It's most probably not monkeypox, but still you should visit a skin specialist. Thank you."
get_model()

@app.route("/predicts", methods = ['GET','POST'])
def predicts():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(r'./static/Data', filename)                    
        file.save(file_path)
        print(filename)
        product = prediction(file_path)
        print(product)
        
    return render_template ('index.html', product = product, user_image =file_path) 


#--- ROUTE -----
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/dailynews", methods=['GET', 'POST'])
def chart1():
    palette_map = ["#97DECE","#62B6B7","#62B6B7","#439A97","#6E7A78","#627A77","#557A75","#497A74","#370617","#03071E"]
    sns.palplot(sns.color_palette(palette_map))

    fig = px.choropleth(data_frame = case,
                    locations="Country",locationmode="country names", color="Confirmed_Cases",
                    color_continuous_scale=palette_map, height=800,scope="world",
                    )
    plt.close() 

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    fig1 = px.choropleth(data_frame = case,
                    locations="Country",locationmode="country names", color="Confirmed_Cases",
                    color_continuous_scale=palette_map,scope="asia",
                    )

    plt.close()

    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    fig2 = px.choropleth(data_frame = case,
                    locations="Country",locationmode="country names", color="Confirmed_Cases",
                    color_continuous_scale=palette_map,scope="africa",
                    )
    plt.close()
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    fig4 = px.choropleth(data_frame = case,
                    locations="Country",locationmode="country names", color="Confirmed_Cases",
                    color_continuous_scale=palette_map,height= 600,scope="europe",
                    )
    plt.close()
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    fig5 = px.choropleth(data_frame = case,
                    locations="Country",locationmode="country names", color="Confirmed_Cases",
                    color_continuous_scale=palette_map,height= 600,scope="north america",
                    )
                    
    plt.close()
    graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
   

    return render_template('dailynews.html',graphJSON=graphJSON , graph1JSON=graph1JSON,  graph2JSON=graph2JSON, graph4JSON=graph4JSON, graph5JSON=graph5JSON )

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

#--- UPLOAD FUNCTION------
@app.route('/upload', methods=['POST']) 
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('index.html', filename=filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)




#--- BTN CAMERA FUNTION --
global capture
capture=0


camera = cv2.VideoCapture(0)
def get_model():
    global model
    model = load_model('model.h5')
    print("Model loaded")

def load_image(img_path):

    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)                   
    img_tensor = np.expand_dims(img_tensor, axis=0)       
    img_tensor /= 255.                                     

    return img_tensor

def prediction(img_path):
    new_image = load_image(img_path)
    
    pred = model.predict(new_image)
    
    print(pred)
    if pred<0.5:
        return "It might be Monkeypox. You should visit a specialist immediately. Thank you."
    else:
        return "It's most probably not monkeypox, but still you should visit a skin specialist. Thank you."
get_model()

def gen_frames():  
    global capture
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['./static/Data', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

@app.route("/", methods=['GET', 'POST'])
@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    if request.method == 'POST':      
        if request.form.get('click') == 'Capture':
            global capture
            capture=1 
        elif request.form.get('click') == 'Capture':
            img = request.files['file'].read()
            npimg = np.fromstring(img, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            product = prediction(img)
            return render_template('index.html', product=product)
            
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')



 





    

if __name__ == '__main__':
    app.run()
    app.debug = (True)
