import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from catboost import Pool, CatBoostClassifier, cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import os
from alphapose.pose import args
from alphapose.pose import cfg
from alphapose.pose import SingleImageAlphaPose
import json
import csv
import pandas as pd
def getdata(all_results):
    csv_results = []
   
    for count,im_res in enumerate(all_results):
        data=[]
        
        for human in im_res['result']:
           
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                data.append(float(kp_preds[n, 0]))
                data.append(float(kp_preds[n, 1]))
                data.append(float(kp_scores[n]))
           
            data.append(float(pro_scores))
        csv_results.append(data)

    df = pd.DataFrame(csv_results)
    return df
import glob
temp_df = pd.read_csv("weights.csv")
class_weight = temp_df[['80', 'class_weight']].to_dict()['class_weight']
model = CatBoostClassifier(
            objective='MultiClass',
            eval_metric='MultiClass',
            num_boost_round=5000,
            class_weights=class_weight,
            learning_rate=0.07964,
            reg_lambda=0.95,
            bootstrap_type='Poisson',
            subsample=0.81,
            max_depth=7, 
            grow_policy='Lossguide',
            min_data_in_leaf=10, 
            max_leaves=152,
            task_type='GPU',
            verbose=0
        )
model.load_model("catboostmodel.cbm")

numberofimagetest=70
outputpath = "testout"
demo = SingleImageAlphaPose(args, cfg)
count=0
activites=["falling forward with hands", "falling forward with knees", "falling backward","fall sideways", "landing sitting on an empty chair","walking", "standing","sitting", "lifting an object",  "jumping","lying down"]

for i,data in enumerate(activites):
    print(str(i)+". "+data)
for imname in glob.glob("test/*.png"):
        image = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2RGB)
        pose = demo.process(imname, image)
        overalldata=[]
        img = demo.getImg()     
        img = demo.vis(img, pose)
        if pose is not None:
            if(len(pose['result'])==1):
                overalldata.append(pose)
                Xtest=getdata(overalldata)
                y_pred = model.predict(Xtest.values)
                for i in y_pred:
                    print(activites[i[0]])
                    if(i[0]<5):
                        print("fall Detected")
                    else:
                        print("Not Fall")
                cv2.imwrite(os.path.join(outputpath, str(count)+".png"), img)
                cv2.imshow("AlphaPose Human Skeleton", img)
                key =cv2.waitKey(1)
        count=count+1
        if(count==numberofimagetest):
            break;

