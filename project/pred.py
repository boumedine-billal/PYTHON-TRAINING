import cv2
import numpy as np
import math
from keras.models import load_model

model = load_model("C:\\Users\\moi\\Desktop\\formation_python_celec\\keras_models\\model01.h5")

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    
    frame = cv2.resize(frame,(256,256))
    
    x = np.zeros((1 , 256, 256, 3), dtype=np.float32)
    
    
    x[0,:,:,:] = frame/255.0
    
    
    
    
    mask = model.predict(x, verbose=1)
    
    mask = mask[0,:,:,0]*255
    
    
    mask = mask.astype(np.uint8)
    
    

    
    # filter
    mask[mask<145] = 0
    
    
    # scale data
    frame = cv2.resize(frame,(500,500))
    mask  = cv2.resize(mask ,(500,500))
    
    
    frame[mask>145] = (153,255,0)
    
    # rectangle
    u1,u2 = frame.shape[:2]
    coins_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in coins_contours if cv2.contourArea(cnt) > 60]
    con=np.zeros((len(large_contours),3),dtype=np.uint32)
    u3,u4 = con.shape[:2]
    if u3 != 0 :
        for contour in large_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    
    cv2.imshow("mask" , mask )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

cap.release()
cv2.destroyAllWindows()

















