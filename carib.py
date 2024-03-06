import tkinter, cv2, time
from PIL import Image, ImageTk
import numpy as np
cam_id = 0

###### 画角サイズの調節 #######
yoko, tate = 1920, 1080
############################

app = tkinter.Tk()
app.title("Python-Camera")
app.geometry(str(yoko)+"x"+str(tate))

canvas1 = tkinter.Canvas(app, width= yoko, height=tate)
canvas1.pack()

carib_dat = []

cap = cv2.VideoCapture(cam_id)

var = tkinter.IntVar()
var.set(0)

def main():
    camera()
    app.mainloop()

def close_window():
    cap.release()
    cv2.destroyAllWindows()
    app.quit()
    print(line_dat)

y_set = int(tate*0.8)
tkinter.Radiobutton(app, value=0, variable=var, text=' マスク ',
                    width=10, height=3).place(x=int(yoko*0.2), y=y_set)

tkinter.Radiobutton(app, value=1, variable=var, text=' カラー ',
                    width=10, height=3).place(x=int(yoko*0.4), y=y_set)

tkinter.Radiobutton(app, value=2, variable=var, text=' グレー ',
                    width=10, height=3).place(x=int(yoko*0.6), y=y_set)

tkinter.Button(app, text='確定して終了', command=close_window, relief="raised",
               width=10, height=3 ).place(x=int(yoko*0.8), y=y_set)


def analysis_blob(binary_img):
    # 2値画像のラベリング処理
    label = cv2.connectedComponentsWithStats(binary_img)
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    max_index = np.argmax(data[:, 4])
    maxblob = center[max_index]
    return maxblob


def camera():
    cap = cv2.VideoCapture(cam_id)
    global line_dat
    time.sleep(0.5)
    ret, frame = cap.read()
    if ret:
        pic = np.array(frame)
        size = pic[:,:,0].shape
        ### Color_to_HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ### Color_detect
        lower = np.array([90,64,0])
        upper = np.array([150,255,255])
        ### Make_mask
        frame_mask = cv2.inRange(hsv, lower, upper)
        dst = cv2.bitwise_and(frame, frame, mask=frame_mask)
        ### Cover_to_Image_by_Mask
        im_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        ### Caputure_Sepalate_2frame
        im_left  = im_gray[:,:int(size[1]/2)]
        im_right = im_gray[:,int(size[1]/2):]
        r_d = analysis_blob(im_right)
        l_d = analysis_blob(im_left)
        r_d, l_d = r_d, l_d
        r_d, l_d = r_d[1], l_d[1]
        line_dat = int( (r_d+l_d)/2 )
        
        if   var.get() == 0 : frame = im_gray
        elif var.get() == 1 : frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif var.get() == 2 : frame = dst
        app.photo = ImageTk.PhotoImage(image = Image.fromarray(frame) )
        canvas1.create_image(0,0, image= app.photo, anchor = tkinter.NW)
        app.after(50, camera)



if __name__=="__main__":
    main()

