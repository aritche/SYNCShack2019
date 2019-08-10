import cv2
import numpy as np
import pyautogui as pag
pag.PAUSE = 0

def nothing(non):
    pass

def largestConvexHull(im, original):
    x,y,w,h = [-1,-1,-1,-1]

    im, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    if (len(contours) != 0):
        maxContour = max(contours, key = cv2.contourArea)
    #for i in range(len(contours)):
    #    hull.append(cv2.convexHull(contours[i], False))
        hull.append(cv2.convexHull(maxContour,False))
    mask = np.zeros((original.shape[0], original.shape[1]))
    for i in range(len(hull)):
        #im = cv2.drawContours(original, contours, i, (0,255,0), 1, 8, hierarchy)
        mask = cv2.drawContours(mask, hull, i, 255, -1, 8)
        x, y, w, h = cv2.boundingRect(hull[i])
        #im = cv2.drawContours(original, hull, i, (255,0,0), 1, 8)

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

            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            im = cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), -1)
            im = cv2.circle(im, (int(centroids[max_label][0]),int(centroids[max_label][1])), 20, (255,0,0), -1)
            cv2.imshow("Biggest component", im)
            cv2.waitKey(1)
            #return [int(centroids[max_label][0]),int(centroids[max_label][1])]
            return [int(x+w/2), int(y)]

    return [-1,-1]

cv2.namedWindow('hand', cv2.WINDOW_NORMAL)
cv2.namedWindow('color', cv2.WINDOW_NORMAL)
cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
cv2.createTrackbar('h_min', 'hand', 0, 179, nothing)
cv2.createTrackbar('h_max', 'hand', 0, 179, nothing)
cv2.createTrackbar('s_min', 'hand', 0, 255, nothing)
cv2.createTrackbar('s_max', 'hand', 0, 255, nothing)
cv2.createTrackbar('v_min', 'hand', 0, 255, nothing)
cv2.createTrackbar('v_max', 'hand', 0, 255, nothing)
cv2.createTrackbar('kernel', 'hand', 0, 30, nothing)
cv2.createTrackbar('kernel_2', 'hand', 0, 30, nothing)
#cv2.createTrackbar('s', 'color', 0, 255, nothing)
#cv2.createTrackbar('v', 'color', 0, 255, nothing)
cv2.createTrackbar('threshold', 'diff', 0, 255, nothing)
cv2.createTrackbar('k', 'diff', 0, 100, nothing)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
curr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
while True:
    ret, frame = cap.read()
    prev = curr
    curr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    t = cv2.getTrackbarPos('threshold', 'diff')
    k = cv2.getTrackbarPos('k', 'diff')
    curr[curr > t] = 255
    curr[curr < 255] = 0
    diff = abs(prev-curr)
    kernel = np.ones((k,k))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

    #bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #edge = cv2.Laplacian(bw, cv2.CV_64F)
    #edge[edge > 50] = 255
    #edge[edge < 255] = 0
    #cv2.imshow('edge', edge)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bgr = frame.copy()

    h_min = cv2.getTrackbarPos('h_min', 'hand')
    h_max = cv2.getTrackbarPos('h_max', 'hand')
    s_min = cv2.getTrackbarPos('s_min', 'hand')
    s_max = cv2.getTrackbarPos('s_max', 'hand')
    v_min = cv2.getTrackbarPos('v_min', 'hand')
    v_max = cv2.getTrackbarPos('v_max', 'hand')
    lower_hand = np.array([59,0,120])
    upper_hand = np.array([137,110,255])
    mask_hand = cv2.inRange(bgr, lower_hand, upper_hand)

    lower = np.array([79,0,0])
    upper = np.array([167,229,65])
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res[res > 0] = 255
    res[res < 255] = 0
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res_hand = cv2.bitwise_and(bgr, bgr, mask=mask_hand)
    res_hand = cv2.cvtColor(res_hand, cv2.COLOR_BGR2GRAY)
    res_hand[res_hand > 0] = 255
    res_hand[res_hand < 255] = 0



    #k = cv2.getTrackbarPos('kernel', 'color')
    #kernel = np.ones((k,k))
    kernel = np.ones((16,16))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((20,20))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    keyboard_mask, dims = largestConvexHull(res, frame)
    key_x, key_y, key_w, key_h = dims

    hand_mask = res_hand

    k = cv2.getTrackbarPos('kernel', 'hand')
    kernel = np.ones((16,16))
    res_hand = cv2.morphologyEx(res_hand, cv2.MORPH_CLOSE, kernel)

    k_2 = cv2.getTrackbarPos('kernel_2', 'hand')
    kernel = np.ones((0,0))
    res_hand = cv2.morphologyEx(res_hand, cv2.MORPH_OPEN, kernel)

    both = 255*((hand_mask/255) * (keyboard_mask/255))
    x, y = getLargestCentroid(np.uint8(both))
    if (x != -1 and y != -1):
        rel_x = (x - key_x) / key_w
        rel_y = (y - key_y) / key_h
        absolute_x = pag.size()[0] - pag.size()[0] * rel_x
        absolute_y = pag.size()[1] * rel_y
        #absolute_y = pag.size()[1] - pag.size()[1] * 
        #pag.moveTo(pag.size()[0]-pag.size()[0]*x/frame.shape[1],pag.size()[1]-pag.size()[1]*y/(frame.shape[0]/4))
        pag.moveTo(absolute_x, absolute_y)

    cv2.imshow('color',keyboard_mask)
    cv2.imshow('hand', hand_mask)
    cv2.imshow('both', both)
    #cv2.imshow('hand',res_hand)
    cv2.waitKey(1)
