import numpy as np
import cv2 as cv
import pandas as pd
import csv
import os

def init(train_data_filename):
    if os.path.exists(train_data_filename):
        if os.path.getsize(train_data_filename):
            for file in os.listdir(train_data_filename)[0:]:
                file_path = train_data_filename + '/' + file
                os.remove(file_path)
    else:
        os.mkdir(train_data_filename)
result=["Result/Go Back","Result/Go Straight","Result/Turn Right","Result/Turn Left"]
wrong=["Wrong/Go Back","Wrong/Go Straight","Wrong/Turn Right","Wrong/Turn Left"]
for i in result:
    init(i)
for j in wrong:
    init(j)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 400,
                       qualityLevel = 0.01,
                       minDistance = 2,
                       blockSize = 5 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (200, 200),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# path='testing/GB/5/Go Back'

# Create some random colors
color = np.random.randint(0, 255, (400, 3))
def constructData(path):
    list = np.arange(1, 21).tolist()
    if path == 'testing/GB/':
        list1 = [1, 7, 11, 17, 24, 41, 45, 60, 70, 84, 91, 95, 102, 109, 129, 143, 148, 157, 175, 180]
        list2 = [6, 8, 15, 20, 40, 44, 59, 67, 80, 88, 94, 101, 108, 122, 142, 145, 154, 168, 179, 183]
    elif path == 'testing/GS/':
        list1=[1,26,61,70,73,140,155,177,201,212,250,273,291,297,327,351,446,457,512,528]
        list2=[25,60,69,72,117,149,176,198,211,239,271,290,295,326,350,445,454,511,515,580]
    elif path == 'testing/TL/':
        list1=[1,12,21,25,36,42,61,56,72,101,131,140,147,175,187,194,200,217,261,292]
        list2=[9,20,24,30,41,55,71,60,94,130,136,144,159,186,193,199,206,223,268,457]
    elif path=='testing/TR/':
        list1=[9,24,34,40,47,114,143,158,163,174,185,200,211,222,228,235,246,252,266,282]
        list2=[23,32,39,46,62,142,149,162,173,181,199,209,221,227,234,245,251,258,281,292]
    df = pd.DataFrame({'start': list1,
                       'finish': list2, },
                      index=list)
    return df
#create dataframe

paths = ['testing/GB/','testing/GS/']



# balance strategy
def BalanceStrategy(good_old,good_new):
    BA=[]
    BB=[]
    BC=[]
    BD=[]
    for i,j in zip(good_old,good_new):
        if i[0]<320:
            if i[1]<240:
                BA.append([(j[0]-i[0]),(j[1]-i[1])])
            else:
                BB.append([(j[0]-i[0]),(j[1]-i[1])])
        else:
            if i[1]<240:
                BC.append([(j[0]-i[0]),(j[1]-i[1])])
            else:
                BD.append([(j[0]-i[0]),(j[1]-i[1])])
    len1=np.linalg.norm(np.array(BA))
    len2=np.linalg.norm(np.array(BB))
    len3=np.linalg.norm(np.array(BC))
    len4=np.linalg.norm(np.array(BD))
    return len1,len2,len3,len4


# detect feature
def FeatureDetect(p2):
    count=0
    for i in p2:
        if i[0][1]>430:
            count=count+1
    return count
#Pre-process
def preProcess(Test):
    TestHSV = cv.cvtColor(Test, cv.COLOR_BGR2HSV)
    mask = cv.inRange(TestHSV, (10, 10, 40), (117, 255, 130))
    imask = mask > 0
    green = np.zeros_like(Test, np.uint8)
    green[imask] = Test[imask]
    r,g,b=cv.split(green)
    return g
# correct
for path in paths:
    correct = 0
    total = 0
    if path == 'testing/GB/':
        temp='/Go Back'
        print("GB")
    elif path == 'testing/GS/':
        temp='/Go Straight'
        print("GS")
    elif path == 'testing/TL/':
        temp='/Turn Left'
        print("TL")
    elif path == 'testing/TR/':
        temp = '/Turn Right'
        print("TR")
    for i in range(1, 20):
        order = i
        # read the image
        spath = path + str(i) + temp
        df=constructData(path)
        for order in range(df.loc[i]["start"], df.loc[i]["finish"]):
            nepath = spath + str(order) + '.png'
            old_frame = cv.imread(nepath, 1)
            # split the rgb image into 3 tube
            r, g0, b = cv.split(old_frame)
             # choose the degree of green and using enhancement techniques
            g=preProcess(old_frame)
            hg = cv.equalizeHist(g)
            old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
            old_gray_H=cv.equalizeHist(old_gray)
            # choose the point p0
            # when using shi-tomasi
            p0 = cv.goodFeaturesToTrack(hg, mask=None, **feature_params)
            # y
            n_p1 = np.arange(1.0, 480.0, 24)
            # x
            n_p2 = np.arange(1.0, 640.0, 32)
            list1 = []
            for i in n_p1:
                for j in n_p2:
                    list1.append([j, i])
            n_p0 = np.array(list1)
            #when using uniform distribution

            # p0= np.reshape(n_p0, (400, 1, 2)).astype(np.float32)
            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

            for i in range(1, 2):
                order = order + 1
                frame = cv.imread(spath + str(order) + '.png')
                frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame_g=preProcess(frame)
                frame_gh=cv.equalizeHist(frame_g)
                frame_gray_hg = cv.equalizeHist(frame_gray)

                # calculate optical flow

                #using pre-process -hg -frame_gh
                # p1, st, err = cv.calcOpticalFlowPyrLK(hg, frame_gh, p0, None, **lk_params)
                # using unprocessed -old_gray_h -frame_gary_hg
                p1, st, err = cv.calcOpticalFlowPyrLK(old_gray_H, frame_gray_hg, p0, None, **lk_params)

                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                    frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                img = cv.add(frame, mask)

                cv.imshow('frame', img)
                cv.imwrite('Result'+temp+'/' + str(order) + str() + '.png', img, )
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break

                # Now update the previous frame and previous points
                # old_gray = frame_gray_h.copy()
                old_gray=frame_gray
                p0 = good_new.reshape(-1, 1, 2)
                len1, len2, len3, len4 = BalanceStrategy(good_old, good_new)
                total = total + 1
                #Using For Go Straight or Go Back
                define = FeatureDetect(p0)

                if temp=='/Go Back':
                    if len1 + len3 >= len2 + len4:
                    # if define>=5:
                        correct = correct + 1
                        cv.imwrite('Result' + temp + '/' + str(order) + str() + '.png', img, )
                    else:
                        cv.imwrite('Wrong' + temp + '/' + str(order) + str() + '.png', img, )
                elif temp=='/Go Straight':
                    # if define<=5:
                    if len1 + len3 <= len2 + len4:
                        correct = correct + 1
                        cv.imwrite('Result' + temp + '/' + str(order) + str() + '.png', img, )
                    else:
                        cv.imwrite('Wrong' + temp + '/' + str(order) + str() + '.png', img, )
                elif temp=='/Turn Left':
                    if len1 + len2 <= len3 + len4:
                        correct = correct + 1
                        cv.imwrite('Result' + temp + '/' + str(order) + str() + '.png', img, )
                    else:
                        cv.imwrite('Wrong' + temp + '/' + str(order) + str() + '.png', img, )
                else:
                    if len1 + len2>=len3 + len4:
                        correct = correct + 1
                        cv.imwrite('Result' + temp + '/' + str(order) + str() + '.png', img, )
                    else:
                        cv.imwrite('Wrong' + temp + '/' + str(order) + str() + '.png', img, )

                    # print(define)
    print('total:' + str(total) )
    print('Correct：'+str(correct))
cv.destroyAllWindows()