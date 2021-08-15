# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 03:20:36 2020

@author: boumedine billal
"""

import numpy as np
import cv2
import math
import re
import time

# *****************************************************************************
def nonlin_numpy(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)
	return 1/(1+np.exp(-x))
def train_dnn_numpy(x,y,nb_hiden,epoc,lr,calback,wwbb,w,b):
    ee = 1.0
    op_n = len(nb_hiden)-2
    error_graph = np.zeros((epoc))
    # randomly initialize our weights with mean 0
    if wwbb:
        w = []
        for i in range(op_n+1):
            w.append(2*np.random.random((nb_hiden[i],nb_hiden[i+1])) - 1)
        b = []
        for i in range(op_n+1):
            b.append(2*np.random.random((1,nb_hiden[i+1])) - 1)
    for j in range(epoc):
        # Feed forward through layers
        c_layers = []
        c_layers.append(x)
        for i in range(op_n+1):
            c_layers.append(nonlin_numpy(np.dot(c_layers[i],w[i])+b[i]))
        e = y - c_layers[op_n+1]
        error = []
        delta = []
        ee = np.mean(np.abs(e))
        error_graph[j] = ee      
        error.append(e)
        delta.append(e * nonlin_numpy(c_layers[op_n+1],deriv=True))
        for i in range(op_n):
            error.append((delta[i]).dot((w[op_n-i]).T))
            delta.append(error[i+1] * nonlin_numpy(c_layers[op_n-i],deriv=True)) 
        for i in range(op_n+1):
            w[i] += (c_layers[i]).T.dot(delta[op_n-i])*lr
            b[i] += (delta[op_n-i]).sum(axis=0)*lr
        ee = round(ee,5)
    return str(ee),w,b
def predict_dnn(x,nb_hiden,w,b):
    k = x
    for i in range(len(nb_hiden)-1):
        k = nonlin_numpy(np.dot(k,w[i])+b[i])
    return k
# *****************************************************************************

font= cv2.FONT_HERSHEY_COMPLEX_SMALL

drawing = False
k = 1
time_ = 0
clok = False
s = ""
bool_lab = False
idid  = 0
la = 0
plot = np.zeros((200,200,3), np.uint8)
bb1 = True 
jump = False
bbj = False
x_train = []
y_train = []

mape = np.zeros((200,200,2))
for i in range(200):
    for j in range(200):
        mape[i,j,:] = [j,i]
mape = (mape.reshape((40000,2)))/200.0

def tre(img,xy,a,color,size):
    if clok:
        kk = int(a/2)
        k = int(math.sqrt(((a**2)*3)/4))
        xy = (xy[0],xy[1]-int(k/2))
        img = cv2.line(img,xy,(xy[0]+kk,xy[1]+k),color,size)
        img = cv2.line(img,xy,(xy[0]-kk,xy[1]+k),color,size)    
        img = cv2.line(img,(xy[0]-kk,xy[1]+k),(xy[0]+kk,xy[1]+k),color,size) 
    return img

def botton(img,xy,wh,name,color):
    img = cv2.rectangle(img,(xy[0],xy[1]),(xy[0]+wh[0],xy[1]+wh[1]),color,1)
    img = cv2.putText(img,name,(xy[0]+int(wh[0]/2)-int(len(name)/2)*9,xy[1]+int(wh[1]/2)+3), font, 0.7,color,1,cv2.LINE_AA)
    return img

def label(img,xy,w,rait,idid2):
    global clok,labels
    img = cv2.rectangle(img,(xy[0],xy[1]),(xy[0]+w,xy[1]+24),(255,255,255),1)
#    if clok and rait and len(s)==0:
#        img = cv2.putText(img,"|",(xy[0]+3,xy[1]+15), font, 0.7,(255,250,250),1,cv2.LINE_AA)
    nm  = int(w/10)
    if rait:
        labels[idid2][3] = s
        lab = labels[idid2][3]
        if len(lab)>nm:
            lab = lab[len(lab)-nm:]
        labels[idid2][5] = lab
        img = cv2.putText(img,lab,(xy[0]+3,xy[1]+15), font, 0.5,(255,250,250),1,cv2.LINE_AA)
        img = cv2.putText(img,labels[idid2][6],(xy[0]-(labels[idid2][7])[0],xy[1]+15-(labels[idid2][7])[1]), font, 0.5,(255,250,250),1,cv2.LINE_AA)
    else:
        if not(labels[idid2][8]):
            img = cv2.putText(img,labels[idid2][5],(xy[0]+3,xy[1]+15), font, 0.5,(255,250,250),1,cv2.LINE_AA)
        else:
            if clok:
                img = cv2.putText(img,labels[idid2][5],(xy[0]+3,xy[1]+15), font, 0.5,(255,250,250),1,cv2.LINE_AA)
        img = cv2.putText(img,labels[idid2][6],(xy[0]-(labels[idid2][7])[0],xy[1]+15-(labels[idid2][7])[1]), font, 0.5,(255,250,250),1,cv2.LINE_AA)
    return img


def draw_circle(event,x,y,flags,param):
    global drawing,bottons,k,labels,bool_lab,s,idid,la,plot,x_train,y_train,bb1

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottons[la][4] = (255,255,255)
        
    # bottons parti
    for j in range(len(bottons)):
        if ((bottons[j][0])[0]<x<(bottons[j][0])[0]+(bottons[j][1])[0]) and ((bottons[j][0])[1]<y<(bottons[j][0])[1]+(bottons[j][1])[1]) :
            bottons[j][6] = True
            bottons[j][4] = (200,200,200)
        else:
            bottons[j][4] = (255,255,255)
    
    # bottons parti
    for j in range(len(bottons)):
        if ((bottons[j][0])[0]<x<(bottons[j][0])[0]+(bottons[j][1])[0]) and ((bottons[j][0])[1]<y<(bottons[j][0])[1]+(bottons[j][1])[1]) and drawing :
            bottons[j][3] = True
            bottons[j][4] = (50,50,50)
            la = j
            # test
            if j==3:
                bottons[4][3] = False
            if j==4:
                bottons[3][3] = False
            if j==0:                
                bb1 = True
            
    # labels parti
    bool1 = False
    ididm = idid
    for i in range(len(labels)): 
        if ((labels[i][0])[0]<x<(labels[i][0])[0]+labels[i][1]) and ((labels[i][0])[1]<y<(labels[i][0])[1]+24) and drawing :
            labels[i][2] = True
            bool_lab = True
            idid = labels[i][4]
            
             
        if not(((labels[i][0])[0]<x<(labels[i][0])[0]+labels[i][1]) and ((labels[i][0])[1]<y<(labels[i][0])[1]+24)) and drawing :
            labels[i][2] = False
        bool1 = bool1 or labels[i][2]
        
    # mouve in label to oder label
    if ididm!=idid:
        s = ""
        s +=labels[idid][3]
    
    # out of all labelse
    if bool1 == False:
        bool_lab = False
        
# ---------------------------- drawing parti
    if 460<x<660 and 50<y<250 and drawing:
        xi = x-460
        yi = y-50
        if bottons[3][3]:
            x_train.append([xi,yi])
            y_train.append([1.0])
            plot = cv2.circle(plot,(xi,yi),2, (153,255,0), -1)
        if bottons[4][3]:
            x_train.append([xi,yi])
            y_train.append([0.0])
            plot = cv2.circle(plot,(xi,yi),2, (230,0,115), -1)
# ----------------------------

def img_main():
    global plot
    img = np.zeros((600,1120,3), np.uint8) 
    img[50:250,460:660] = plot
    for  i in range(len(y_train)):
        if (y_train[i])[0]==1:
            plot = cv2.circle(plot,(int((x_train[i])[0]),int((x_train[i])[1])),2, (153,255,0), -1)
        if (y_train[i])[0]==0:
            plot = cv2.circle(plot,(int((x_train[i])[0]),int((x_train[i])[1])),2, (230,0,115), -1)
    img = cv2.putText(img,"PLOTE SETING:",(975,65), font, 0.7,(255,250,250),1,cv2.LINE_AA)
    img = cv2.line(img,(0   ,40  ),(1200,40  ),(255,250,250),1)
    img = cv2.rectangle(img,(20,50),(420,350),(255,255,255),1)
    img = cv2.rectangle(img,(460,50),(660,250),(255,255,255),1)
    #img = tre(img,(300,300),300,(102, 0, 255),2)
    # creat bottons
    for i in range(len(bottons)):
        img = botton(img,bottons[i][0],bottons[i][1],bottons[i][2],bottons[i][4])
    # creat labels
    for i in range(len(labels)):
        img = label(img,labels[i][0],labels[i][1],labels[i][2],labels[i][4])
    return img


cv2.namedWindow('Dark Net')
cv2.setMouseCallback('Dark Net',draw_circle)




"""          x   y     w   h                                               """
bottons = [[(980 ,380),(120,30 ),'Training'     ,False,(255,255,255),0,False],
           [(980 ,420),(120,30 ),'Stop Train'   ,False,(255,255,255),1,False],
           [(980 ,480),(120,30 ),'Clear'        ,False,(255,255,255),2,False],
           [(975 ,80 ),(60 ,30 ),'class1'       ,False,(255,255,255),3,False],
           [(1045,80 ),(60 ,30 ),'class2'       ,False,(255,255,255),4,False],
           [(10  ,10 ),(30 ,23 ),'x '           ,False,(255,255,255),5,False]]

k1 = (0,25)
"""          x   y    w   down  rame    lab                          vibration"""
labels  = [[(20 ,380),350,False,""   ,0,""   ,"-hiden layers:"   ,k1,False    ],
           [(20 ,430),120,False,""   ,1,""   ,"-learning rate:"  ,k1,False    ],
           [(20 ,480),400,False,""   ,2,""   ,"-Eroors messages:",k1,True     ]]


def predict_plot(nb_hiden,wo,bo):
    global mape
    plot_f = np.zeros((200,200,3), np.uint8)
    y = predict_dnn(mape,nb_hiden,wo,bo)
    y = y.reshape((200,200))
    plot_f[:,:,0] = ((1/(1+np.exp(-60*(y-0.5))))*255).astype(np.uint8)
    plot_f[:,:,2] = ((1/(1+np.exp( 60*(y-0.5))))*255).astype(np.uint8)
    return plot_f
    
###############################################################################
# a.insert(0,15)
def espasse(s):
    return [int(sr) for sr in s.split() if sr.isdigit()]
        
w = 400
h = 300
size_node = 3
size_m_w = w-100
size_m_h = h-100
img = np.zeros((h,w,3), np.uint8)
frame = np.zeros((600,1120,3), np.uint8) 

while(1):
# *****************************************************************************
    # training parti
    if bottons[0][3] == True:
        labels[2][5] = "" 
        labels[2][3] = ""
        str1 = espasse(labels[0][3])
        lrr  = (re.findall("\d+\.\d+",labels[1][3]))
        if (len(str1)!=0) and (len(lrr)!=0) and (len(y_train)!=0):
            str1.insert(0,2)
            str1.append(1)
            ###############################################################################
            if bb1 == True:
                nb_hiden = str1
                epock = 100
                lr = float(lrr[0])
                calback = 20
            
                 
                nb = len(nb_hiden)
                l1 = int(size_m_w/nb)
                l2 = int(l1-(size_node))
                
                y1 = []
                for i in range(nb):
                   y1.append(int(size_m_h/2-nb_hiden[i]*5+10))  # 20 -- 5    
                   
                xt = (np.array(x_train))/200
                yt = np.array(y_train)
                _,wo,bo = train_dnn_numpy(xt,yt,nb_hiden,0,lr,calback,True,0,0) ####
            # *****************************************************************************            
            
            epoc = int(epock-epock/((i+1)))
            #break
            ee,wo,bo = train_dnn_numpy(xt,yt,nb_hiden,epoc,lr,calback,False,wo,bo) ####
            w_net = []
            for i in range(nb-1):
                w_net.append(wo[i])
            img = np.zeros((h,w,3), np.uint8)
            pose = []
            for i in range(nb):
                list_t = []
                for j in range(nb_hiden[i]):
                    xx = (l2+size_node)*i+50
                    yy = y1[i]+j*10+50  # 40 -- 10
                    img = cv2.circle(img,(xx,yy), size_node, (153,255,0), -1)
                    list_t.append([[xx-size_node,yy],[xx+size_node,yy]])
                pose.append(list_t)
            # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww_net preparation   
            maxe = -100        
            for i in range(nb-1):
                mp = w_net[i]
                m = (mp).max()
                if m > maxe:
                    maxe = m
            for i in range(nb-1):
                w_net[i] = (w_net[i]/maxe)*255
                (w_net[i])[w_net[i]<0] = 0
                w_net[i] = (w_net[i]).astype(np.uint8) 
            w_net2 = []
            for i in range(nb-1):
                w_net2.append(wo[i])
            mine = 100        
            for i in range(nb-1):
                mp = w_net2[i]
                m = (mp).min()
                if m < mine:
                    mine = m
            for i in range(nb-1):
                w_net2[i] = (w_net2[i]/mine)*255
                (w_net2[i])[w_net2[i]<=0] = 0
                w_net2[i] = (w_net2[i]).astype(np.uint8)    
            for i in range(nb-1):
                for j in range(nb_hiden[i]):
                    for k in range(nb_hiden[i+1]):
                        c = int((w_net[i])[j,k])
                        if c > 0:
                            color = (0,c,0)
                        else:
                            c = int((w_net2[i])[j,k])
                            color = (0,0,c)
                        img = cv2.line(img,(pose[i][j][1][0],pose[i][j][1][1]),(pose[i+1][k][0][0],pose[i+1][k][0][1]),color,2)     
            img = cv2.putText(img,"ERROR : "+ee,(int(w/2)-80,h-20), font, 0.7,(255,250,250),1,cv2.LINE_AA)
            bb1 = False
            if jump == True :    
                plot = predict_plot(nb_hiden,wo,bo)
        else:
            labels[2][5] = "Error: presse clear."
# *****************************************************************************    



    
###############################################################################
    if bottons[2][3] == True:
        bottons[0][3] = False
        labels[0][5] = ""  
        labels[1][5] = "" 
        labels[2][5] = "" 
        labels[2][3] = "" 
        labels[0][3] = ""  
        labels[1][3] = "" 
        x_train = []
        y_train = []

        plot = np.zeros((200,200,3), np.uint8)
        bottons[2][3] = False
             
    if bottons[5][3] == True:
        break
    
    if bottons[1][3] == True:
        bottons[0][3] = False
        bottons[1][3] = False
        
        t1 = time.time()
        plot = predict_plot(nb_hiden,wo,bo)
        t2 = time.time()
        print(t2-t1)
        
             
        #del wo,bo
        x_train = []
        y_train = []

 
###############################################################################
    jump = False
    time_ += 1
    if time_%7==0:
        clok = not(clok)
        if time_==1000:
            time_=1
        if clok == True and bbj == False :
            jump = True
        bbj = clok
            
    if bool_lab==True:
        while(True):
            k = cv2.waitKey(1)
            if (k != -1)or (bool_lab==False):
                break
        if k != -1:
            if k == 8:
                s = s[0:len(s)-1]
            else:
                if 32<=k<=176:
                    s += chr(k)
###############################################################################
    frame = img_main()
    frame[50:350,20:420] = img
    cv2.imshow('Dark Net',frame)   
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break
    
cv2.destroyAllWindows()

