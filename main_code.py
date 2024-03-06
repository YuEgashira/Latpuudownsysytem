# -*- coding: utf-8 -*-
import subprocess as sb ; import numpy as np ; from tqdm import tqdm
from datetime import datetime, date ; from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import scipy, cv2, time, csv
from scipy import interpolate
astrisk = "****************************************"
haihuns = "----------------------------------------"
#######################################
path = "/Users/shiozawalab/Documents/train_blue/"
cam_id = 0»
#######################################

### 何回するのか #######
kaisuu = 3
######################

def main():
    ############################################################
    line = np.zeros(2)
    print(haihuns+'\n ____Calibration_Start____ \n'+haihuns)
    line[0] = carib()
    print(astrisk+'\n ____Line1_Setting_OK____ \n'+astrisk)
    time.sleep(0.5)
    line[1] = carib()
    print(astrisk+'\n ____Line2_Setting_OK____ \n'+astrisk)
    time.sleep(1)
    #cam_reset()
    ############################################################
    fname = fname_make()
    
    print(haihuns+'\n ____Training_1st_set_Start____ \n'+haihuns)
    name = fname+"_1"
    train(line, name)
    print(haihuns+'\n ____Training_1st_set_Finish____ \n'+haihuns)
    time.sleep(1)
    ############################################################
    
    
    
    ############################################################
    
    print(haihuns+'\n ____Training_2nd_set_Start____ \n'+haihuns)
    name = fname+"_2"
    train(line, name)
    print(haihuns+'\n ____Training_2nd_set_Finish____ \n'+haihuns)
    time.sleep(2)
    ############################################################
    
    print(haihuns+'\n ____Training_3rd_set_Start____ \n'+haihuns)
    name = fname+"_3"
    train(line, name)
    print(haihuns+'\n ____Training_3rd_set_Finish____ \n'+haihuns)
    time.sleep(2)
    ############################################################
    #fname = ""
    #train_analy(fname)
    #"""

def train_analy(fname):
    fname = "2022_10_14_1416"
    print("Done")
    file_1 = path+'train_csv/'+fname+'_1.csv'
    file_2 = path+'train_csv/'+fname+'_2.csv'
    file_3 = path+'train_csv/'+fname+'_3.csv'
    dat = np.genfromtxt(file_1, skip_header=0, delimiter=",")
    past = dat[:,0]
    
    lines = np.arange( min(dat[:,0]) , max(dat[:,0]), 0.1)
    r_p = interpolate.interp1d( dat[:,0], dat[:,1] , kind="quadratic")(lines)
    rigt = interpolate.interp1d( dat[:,0], dat[:,2] , kind="quadratic")(lines)
    left = interpolate.interp1d( dat[:,0], dat[:,3],  kind="quadratic")(lines)
    
    plt.figure("test")
    plt.subplot(311)
    plt.title("RED_POINT")
    plt.ylim(700,0)
    plt.plot(past, dat[:,1],"r.")
    plt.plot(lines, r_p, "b-")
    
    plt.subplot(312)
    plt.title("RIGHT")
    plt.ylim(700,0)
    plt.plot(past, dat[:,2],"r.")
    plt.plot(lines, rigt, "b-")
    plt.plot( [1,19], [min(r_p),min(r_p)],"g-" )
    plt.plot( [1,19], [max(r_p),max(r_p)],"g-" )
    plt.subplot(313)
    
    plt.title("LEFT")
    plt.ylim(700,0)
    plt.plot( [1,19], [min(r_p),min(r_p)],"g-" )
    plt.plot( [1,19], [max(r_p),max(r_p)],"g-" )
    plt.plot(past, dat[:,3],"r.")
    plt.plot(lines, left, "b-")
    plt.show()
    


def cam_reset():
    cap, i = cv2.VideoCapture(0), 0
    while True:
        ret, frame = cap.read()
        cv2.imshow('camera' , frame)
        i = i+1
        if i>10 : break
    cap.release()
    cv2.destroyAllWindows()

def carib():
    ret = sb.run(["python", path+'carib.py'],capture_output=True,text=True)
    #""""
    ret = str(ret)
    print(ret)
    cut = ret[ret.find("returncode"):ret.find("stderr")]
    cut = cut.replace("n",",")
    cut = cut.replace("=",",")
    cut = cut.replace('\\','')
    cut = cut.split(",")
    print(cut)
    dat = int(cut[4][1:])
    return dat

def analysis_blob(binary_img):
    label = cv2.connectedComponentsWithStats(binary_img)
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    max_index = np.argmax(data[:, 4])
    maxblob = center[max_index]
    return maxblob

def fname_make():
    d_today = str( date.today() ).replace("-",'_')
    dt_now = datetime.now()
    hour, minit = str(dt_now.hour), str(dt_now.minute)
    if len(hour) ==1 : hour  = str(0) + hour
    if len(minit) ==1 : minit = str(0) + minit
    fname = d_today+'_'+hour+minit
    return fname

def train(line, fname):
    line_1,line_2 = int(line[0]),int(line[1])
    sta_t = time.time()
    imgArr = []
    cv2.namedWindow("Training", cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(cam_id)
    while (True):
        ### Time_Setting
        end_t = time.time()
        past = round(end_t-sta_t,2)
        
        ### Camera_ON
        ret, img = cap.read()
        pic = np.array(img)
        size = pic[:,:,0].shape
        
        ### Line_Draw_to_IMG
        zou = int((line_2-line_1)/2)
        cv2.line(img, ( 0, line_1), ( size[1], line_1), ( 0, 0, 255), thickness=10, lineType=cv2.LINE_AA)
        cv2.line(img, ( 0, line_2), ( size[1], line_2), ( 0, 0, 255), thickness=10, lineType=cv2.LINE_AA)
        
        ### Count_Down
        if past<1:
            time.sleep(0.1)
            ### Count_Down_Text
            cv2.putText(img, "Wait", (int(size[0]/2), int(size[1]/2)),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 5, cv2.LINE_AA)
        
        elif past<6:
            time.sleep(0.2)
            ### Count_Down_Time
            count = str(6-int(past))
            ### Count_Down_Text
            cv2.putText(img, "Count_Down: "+count, (int(size[0]/2), int(size[1]/2)),
                        cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 5, cv2.LINE_AA)
        
        ### Train_Start
        elif past >=6:
            #time.sleep(0.09)
            ### red_Point_Move
            point = int( -1*np.cos(np.pi* 1/3*past )*zou +zou+line_1 )
            cv2.circle(img, ( 40, point), 20, ( 0, 0, 255), thickness=-1)
            cv2.circle(img, ( size[1]-40, point), 20, ( 0, 0, 255), thickness=-1)
            ### Blue_Detect
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_min = np.array([90,64,0])
            hsv_max = np.array([150,255,255])
            mask = cv2.inRange(hsv, hsv_min, hsv_max)
            dst = cv2.bitwise_and(img, img, mask=mask)
            ### Gray_Only
            im_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            ### Max_Area_pickup
            r_d = analysis_blob( im_gray[:,size[0]:] )
            l_d = analysis_blob( im_gray[:,:size[0]] )
            r_d, l_d = r_d[1], l_d[1]
            ### Movie_Use
            imgArr.append(img)
            ### Locate_Bar_to_CSV
            result = [ past-5, point, r_d, l_d ]
            with open(path+'train_csv/'+fname+'.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(result)
        
        ### Show_IMAGE
        cv2.imshow("Training", img)
        if past>6+6*kaisuu or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    all_time = int( (6*kaisuu) )
    all_date = len(imgArr)
    fps = int(all_date/all_time)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = None
    video_name = path+'train_movie/'+fname+'.mp4'
    pbar = tqdm(desc=fname+"_Video_Writing ...", postfix="dict", total=all_date)
    for img in imgArr:
        pbar.update(1)
        if(video is None):
            h, w, _ = img.shape
            video = cv2.VideoWriter( video_name , fourcc, fps, (w,h))
        video.write(img)
    video.release()
    print(astrisk+"\n Movie_fps : "+str(fps)+"\n"+astrisk)
    pbar.close()






if __name__=="__main__":
    main()