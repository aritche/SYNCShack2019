import cv2
import numpy as np

def nothing(non):
    pass

def largestConvexHull(im, original):
    im, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    if (len(contours) != 0):
        maxContour = max(contours, key = cv2.contourArea)
    #for i in range(len(contours)):
    #    hull.append(cv2.convexHull(contours[i], False))
        hull.append(cv2.convexHull(maxContour,False))
    for i in range(len(hull)):
        #im = cv2.drawContours(original, contours, i, (0,255,0), 1, 8, hierarchy)
        im = cv2.drawContours(original, hull, i, (255,0,0), 1, 8)

    return im


cv2.namedWindow('hand', cv2.WINDOW_NORMAL)
cv2.namedWindow('color', cv2.WINDOW_NORMAL)
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

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos('h_min', 'hand')
    h_max = cv2.getTrackbarPos('h_max', 'hand')
    s_min = cv2.getTrackbarPos('s_min', 'hand')
    s_max = cv2.getTrackbarPos('s_max', 'hand')
    v_min = cv2.getTrackbarPos('v_min', 'hand')
    v_max = cv2.getTrackbarPos('v_max', 'hand')
    lower_hand = np.array([h_min,s_min,v_min])
    upper_hand = np.array([h_max,s_max,v_max])
    mask_hand = cv2.inRange(hsv, lower_hand, upper_hand)

    lower = np.array([79,0,0])
    upper = np.array([167,229,65])
    mask = cv2.inRange(hsv, lower, upper)

    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res[res > 0] = 255
    res[res < 255] = 0
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    res_hand = cv2.bitwise_and(hsv, hsv, mask=mask_hand)
    res_hand = cv2.cvtColor(res_hand, cv2.COLOR_BGR2GRAY)
    res_hand[res_hand > 0] = 255
    res_hand[res_hand < 255] = 0



    #k = cv2.getTrackbarPos('kernel', 'color')
    #kernel = np.ones((k,k))
    kernel = np.ones((16,16))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((20,20))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    component = largestConvexHull(res, frame)


    k = cv2.getTrackbarPos('kernel', 'hand')
    kernel = np.ones((k,k))
    res_hand = cv2.morphologyEx(res_hand, cv2.MORPH_CLOSE, kernel)

    k_2 = cv2.getTrackbarPos('kernel_2', 'hand')
    kernel = np.ones((k_2,k_2))
    res_hand = cv2.morphologyEx(res_hand, cv2.MORPH_OPEN, kernel)


    cv2.imshow('color',component)
    cv2.imshow('hand',res_hand)
    #cv2.imshow('hand',res_hand)
    cv2.waitKey(1)
