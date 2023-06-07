def get_x(row): 
    return row['image_id']

import os
import json
from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
from fastai.vision.all import *

app = Flask(__name__)

learn = load_learner('new_own_vgg.pkl')
interp = ClassificationInterpretation.from_learner(learn)

def get_y(row): 
    return row['label']

def classify_image(img):
    mapping = {1: 'chickenpox',
 2: 'corns',
 3: 'eczema',
 4: 'monkeypox',
 5: 'normal',
 6: 'warts'}
    print(img)
    pred_idx,_,probs = learn.predict(img)
    pred_class = mapping[int(pred_idx)]
    prob = probs.max()
    prob = prob.numpy()
    return pred_class,prob

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def upload_image():
    option = request.form["option"]
    if option == 'Resnet152':
        learn = load_learner('new_own_vgg.pkl')
    uploaded_file = request.files["image"]
    img_bytes = uploaded_file.read()
    image = Image.open(BytesIO(img_bytes))
    saved_file_path = os.path.join("tempDir", uploaded_file.filename)
    with open(saved_file_path, "wb") as f:
        f.write(img_bytes)
    pred_class, prob = classify_image(saved_file_path)
    description_file = open('comment.json')
    disease_details = json.load(description_file)
    cancers = ['Basal cell carcinoma', 'Melanoma', 'Squamous cell carcinoma']
    if pred_class in cancers:
        result = "Skin Cancer Detected. Please consult a doctor immediately."
    elif pred_class == "Normal":
        result = "No Skin disease found. Please consult a doctor if you have any concerns."
    else:
        result = "This is a Non-Cancerous Skin disease. Please consult a doctor for further treatment."
    if pred_class != "Normal":
        description = disease_details[pred_class]["Description"]
        symptoms = disease_details[pred_class]["Symptoms"]
        causes = disease_details[pred_class]["Causes"]
        risk_factors = disease_details[pred_class]["Risk Factors"]
        treatment = disease_details[pred_class]["Treatment"]
        diagnosis = disease_details[pred_class]["Diagnosis"]
        treatment = disease_details[pred_class]["Treatment"]
    return render_template(
        "index.html", 
        prediction_text=f"The model predicts the image as: {pred_class} with probability: {prob*100:.2f}%",
        result=result,
        description=description,
        symptoms=symptoms,
        causes=causes,
        risk_factors=risk_factors,
        diagnosis=diagnosis,
        treatment=treatment
    )

if __name__ == "__main__":
    app.run(debug=True)
