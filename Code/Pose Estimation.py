import cv2
import os
from alphapose.pose import args
from alphapose.pose import cfg
from alphapose.pose import SingleImageAlphaPose
import json
import csv
import pandas as pd

def writecsvFile(all_results,tags):
    csv_results = []
   
    for count,im_res in enumerate(all_results):
        data=[]
        im_name = im_res['imgname']
        data.append(im_name)
        for human in im_res['result']:
           
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                data.append(float(kp_preds[n, 0]))
                data.append(float(kp_preds[n, 1]))
                data.append(float(kp_scores[n]))
           
            data.append(float(pro_scores))
        data.append(tags[count])
        csv_results.append(data)

    df = pd.DataFrame(csv_results)
    df.to_csv('Hu_sk_dataset.csv')
   
def test():
    outputpath = "result"
    demo = SingleImageAlphaPose(args, cfg)
    count=0
    overalldata=[]
    dataset = pd.read_csv("finaldataset.csv")
    image=dataset['Image'].values
    label=dataset['Target'].values
    tags=[]
    for imname in image:

        image = cv2.cvtColor(cv2.imread(imname), cv2.COLOR_BGR2RGB)
        pose = demo.process(imname, image)
        img = demo.getImg()     
        img = demo.vis(img, pose)  
        cv2.imwrite(os.path.join(outputpath, str(count)+".png"), img)
        cv2.imshow("AlphaPose Human Skeleton", img)
       
        key =cv2.waitKey(1)
        if pose is not None:
            if(len(pose['result'])==1):
                overalldata.append(pose)
                tags.append(label[count])
        count=count+1
    writecsvFile(overalldata,tags)

if __name__ == "__main__":
    test()

