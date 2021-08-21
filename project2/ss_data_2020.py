
import numpy as np
import cv2


import os
import glob




drawing  = False 
drawing2 = False 

exite_ = False
next_  = False
back_  = False
save_  = False
mask_  = False

font = cv2.FONT_HERSHEY_SIMPLEX
bo = True
alfa = 0

kernel = np.ones((10,10),np.uint8)

tx  = "train\\x\\x"
ty  = "train\\y2\\y"

img_dir = "train\\x"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

nb = len(files)   #####            ##########           ###########           ############    ###########  ♦  !!!!       !!!! !!!


def draw_circle(event,x,y,flags,param):
    global drawing , next_ ,exite_,back_,save_,mask_,bo
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
            
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
    
    if 520<x<590 and 10<y<40 and drawing == True:
        exite_ = True
        
    if 520<x<590 and 60<y<90 and drawing == True:
        next_ = True
        
    if 520<x<590 and 110<y<140 and drawing == True:
        back_ = True
        
    if 520<x<590 and 160<y<190 and drawing == True:
        save_ = True
        
    if 520<x<590 and 210<y<240 and drawing == True:
        mask_ = True
        
        
    if 512<x<600 and drawing == True :
        bo = True

        


def draw_circle2(event,x,y,flags,param):
    global drawing2,bo,xb,yb
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing2 = True
            
    if event == cv2.EVENT_LBUTTONUP:
        drawing2 = False

    if drawing2 == True and 0<x<600:
        print(bo)
        if bo == True:
            xb,yb = x,y
            bo = False
        cv2.line(log_main,(xb,yb),(x,y),(255,0,0),2)
        cv2.line(log_main,(xb+600,yb),(x+600,y),(255,0,0),2)
        xb,yb = x,y


def resize_b(image):
    a = 600
    img_x = np.ones((a,a,3), np.uint8)
    img_x[:,:,0] = cv2.resize(image[:,:,0],(a,a))
    img_x[:,:,1] = cv2.resize(image[:,:,1],(a,a))
    img_x[:,:,2] = cv2.resize(image[:,:,2],(a,a))
    return img_x

def resize_c(image):
    a = 256
    img_x = np.ones((a,a,3), np.uint8)
    img_x[:,:,0] = cv2.resize(image[:,:,0],(a,a))
    img_x[:,:,1] = cv2.resize(image[:,:,1],(a,a))
    img_x[:,:,2] = cv2.resize(image[:,:,2],(a,a))
    return img_x

def x_to_main(img_x,img_main):
    img_main[0:600,0:600,:] = img_x
    return img_main

def y_to_main(img_y,img_main):
    img_main[0:600,600:1200,:] = img_y
    return img_main

def reg_main():
    img_main = np.zeros((256,600,3), np.uint8)#512
    img_main[10:40,520:590,2]=255
    img_main[:,513:515,1]=255
    return img_main

def cta_main(img_main):
    img_main[:,515:600,:]=(0,0,0)
    img_main[10:40,520:590,:]=(0,0,255)
    img_main[:,513:515,1]=255
    
    img_main[60:90,520:590,:  ]=255
    cv2.putText(img_main,'next',(520,85), font, 1,(255,0,0),1,cv2.LINE_AA)
    
    img_main[110:140,520:590,:]=255
    cv2.putText(img_main,'back',(520,135), font, 1,(255,0,0),1,cv2.LINE_AA)
    
    img_main[160:190,520:590,:]=255
    cv2.putText(img_main,'save',(520,185), font, 1,(255,0,0),1,cv2.LINE_AA)
    
    img_main[210:240,520:598,:]=255
    
 
    return img_main

  
img_x = np.zeros((256,256,3), np.uint8)
img_y = np.zeros((256,256,3), np.uint8)
img_main = reg_main()

log_main  = np.zeros((600,1200,3),dtype=np.uint8)

cv2.namedWindow('image')
cv2.namedWindow('log_main')

cv2.setMouseCallback('image',draw_circle)
cv2.setMouseCallback('log_main',draw_circle2)


ind = 0

while(1):
    


    if next_ == True and alfa < nb: 
        alfa += 1 
        sr = tx+str(alfa)+".png"
        image = cv2.imread(sr)
        image = resize_b(image)
        log_main = x_to_main(image,log_main)  

        log_main[0:600,600:1200,:] = 0
        next_ = False  
        
        
        img_main[0:256,0:256,:] = 0
        cv2.putText(img_main,str(alfa),(20,200), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,200,101),2,cv2.LINE_AA)

    if back_ == True and alfa > 1 :
        alfa -= 1 
        image = cv2.imread(tx+str(alfa)+".png")
        image = resize_b(image)
        log_main = x_to_main(image,log_main)  

        log_main[0:600,600:1200,:] = 0
        back_ = False
        
        
        img_main[0:256,0:256,:] = 0
        cv2.putText(img_main,str(alfa),(20,200), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,200,101),2,cv2.LINE_AA)

    if save_ == True : 
        #cv2.imwrite(tx+str(ind)+".png",image                    )
        sr = ty+str(alfa)+".png"
        cv2.imwrite(sr,resize_c(log_main[0:600,600:1200,:]))
        ind += 1
        save_ = False
        print(alfa,"saved")
        
    if mask_ == True : 
        im_in = log_main[0:600,600:1200,0]
        
        th, im_th = cv2.threshold(im_in, 127, 255, cv2.THRESH_BINARY)

# Copy the thresholded image
        im_floodfill = im_th.copy()

# Mask used to flood filling.
# NOTE: the size needs to be 2 pixels bigger on eac6h side than the input image
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

# Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground
        log_main[0:600,600:1200,0] = im_th | im_floodfill_inv
        
        mask_ = False
        
        
        bo=True

    if exite_ == True :
        break

        
        

    cv2.imshow('image',img_main)
    cv2.imshow('log_main',log_main)
    img_main = cta_main(img_main)
    cv2.putText(img_main,str(alfa),(520,235), font, 1,(255,0,0),1,cv2.LINE_AA)
   
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break


    
cv2.destroyAllWindows()

