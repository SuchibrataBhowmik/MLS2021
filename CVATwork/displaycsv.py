import os
import cv2
import pandas as pd

def display(data):
    
    for i in range (len(os.listdir("frames/"))):
        path = "frames/frame_"+ "{:06}".format(i) +".PNG"
        image = cv2.imread(path)

        mask = data['frameno'].values == i
        df = data.loc[mask]
        
        if df['outside'].any()==1:
            continue
        
        car1 = df[ df['label'].values == "car_1" ]
        car2 = df[ df['label'].values == "car_2" ]
        bike = df[ df['label'].values == "bike" ]
        
        if not car1.empty:
            cv2.rectangle(image,(car1['left'],car1['top']),(car1['right'],car1['bottom']),(0,255,0),1)
            cv2.putText(image,'1',(car1['left'],car1['top']),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
        if not bike.empty:
            cv2.rectangle(image,(bike['left'],bike['top']),(bike['right'],bike['bottom']),(0,0,255),1)
            cv2.putText(image,'2',(bike['left'],bike['top']),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255))
        if not car2.empty:
            cv2.rectangle(image,(car2['left'],car2['top']),(car2['right'],car2['bottom']),(255,0,0),1)
            cv2.putText(image,'3',(car2['left'],car2['top']),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
        
        cv2.imshow("video",image)
        cv2.waitKey(50)
    
    cv2.destroyAllWindows()        
    return

data = pd.read_csv('track.csv')
display(data)




