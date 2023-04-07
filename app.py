import numpy as np
from PIL import Image
from DensenetMod import FeatureExtractorD
from EfficientNetMod import FeatureExtractorE
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from time import time
import pandas as pd

app = Flask(__name__)

# Read image features
DenseFeature = FeatureExtractorD()
EfficientFeature = FeatureExtractorE()

DenseEffDF=[]

img_paths = []

for feature_path in Path("./static/DenseEffNetFeatures").glob("*.npy"):
    DenseEffDF.append(np.load(feature_path))
DenseEffDF = np.array(DenseEffDF)



for img in  Path("./static/image").glob("*.jpg"):
    img_paths.append(img)

def extract_data(isic_id):
    # Load data from CSV file
    df = pd.read_csv('data.csv')
    
    # Extract data based on ISIC ID
    data = df.loc[df['isic_id'] == isic_id]
    
    return data.to_dict('records')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start = time()
        file = request.files['query_img']
        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        queryA = DenseFeature.extract(img)
        queryB=EfficientFeature.extract(img)
        query=np.concatenate((queryA,queryB))
        dists = np.linalg.norm(np.subtract(DenseEffDF,query), axis=1)  # L2 distances to features
        # Run search
        ids = np.argsort(dists)[1:10]  # Top 10 results
        scores = [(dists[id], img_paths[id]) for id in ids]  
        skinRecords=[]
        for score in scores:
            pathname=score[1].stem.split('_')
            isic_id=pathname[0]+"_"+pathname[1]
            record=extract_data(isic_id)
            
            skinRecords.append(record)
        speed = time()-start
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               skinRecords=skinRecords)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0",port=5003,debug=True)
