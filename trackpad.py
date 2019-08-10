import cv2
import numpy as np
import pyautogui as pag
import time
pag.PAUSE = 0

def nothing(non):
    pass

def largestConvexHull(im, original):
    x,y,w,h = [-1,-1,-1,-1]

    im, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    if (len(contours) != 0):
        maxContour = max(contours, key = cv2.contourArea)
        hull.append(cv2.convexHull(maxContour,False))
    mask = np.zeros((original.shape[0], original.shape[1]))
    for i in range(len(hull)):
        mask = cv2.drawContours(mask, hull, i, 255, -1, 8)
        x, y, w, h = cv2.boundingRect(hull[i])

    return [mask, [x,y,w,h]]

def getLargestCentroid(im):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im, connectivity=4)
    if (nb_components > 1):
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        if (max_size > 80):
            masked = im.copy()
            masked[output == max_label] = 255
            masked[output != max_label] = 0
            masked, dims = largestConvexHull(masked,masked)
            x, y, w, h = dims

            #im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            #im = cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), -1)
            #im = cv2.circle(im, (int(centroids[max_label][0]),int(centroids[max_label][1])), 20, (255,0,0), -1)
            return [int(x+w/2), int(y)]

    return [-1,-1]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

HIST_LIMIT = 1
hist_x = []
hist_y = []
#fps = []
while True:
    #start_time = time.time()

    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_hand = np.array([59,0,120])
    upper_hand = np.array([137,110,255])
    mask_hand = cv2.inRange(frame, lower_hand, upper_hand)

    lower = np.array([79,0,0])
    upper = np.array([167,229,65])
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res[res > 0] = 255
    res[res < 255] = 0
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res_hand = cv2.bitwise_and(frame, frame, mask=mask_hand)
    res_hand = cv2.cvtColor(res_hand, cv2.COLOR_BGR2GRAY)
    res_hand[res_hand > 0] = 255
    res_hand[res_hand < 255] = 0

    kernel = np.ones((16,16))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((20,20))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    keyboard_mask, dims = largestConvexHull(res, frame)
    key_x, key_y, key_w, key_h = dims

    both = 255*((res_hand/255) * (keyboard_mask/255))
    x, y = getLargestCentroid(np.uint8(both))
    if (x != -1 and y != -1):
        rel_x = (x - key_x) / key_w
        rel_y = (y - key_y) / key_h
        absolute_x = pag.size()[0] - pag.size()[0] * rel_x
        absolute_y = pag.size()[1] * rel_y
        hist_x.append(absolute_x)
        hist_y.append(absolute_y)
        if (len(hist_x) > HIST_LIMIT):
            hist_x.pop(0)
            hist_y.pop(0)
        pag.moveTo(sum(hist_x)/len(hist_x), sum(hist_y)/len(hist_y))
        #pag.moveTo(absolute_x, absolute_y)

    #fps.append(1.0 / (time.time() - start_time))
    #print("FPS: ", 1.0 / (time.time() - start_time))
    #print(sum(fps) / len(fps))
