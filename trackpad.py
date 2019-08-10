import cv2
import numpy as np
import pyautogui as pag
import time
import matplotlib.pyplot as plt

pag.PAUSE = 0
pag.FAILSAFE = False

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

            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            im = cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), -1)
            im = cv2.circle(im, (int(centroids[max_label][0]),int(centroids[max_label][1])), 20, (255,0,0), -1)
            return [int(x+w/2), int(y)]

    return [-1,-1]

cap = cv2.VideoCapture(0)
ret, curr = cap.read()

HIST_LIMIT = 2
DIFF_LIMIT = 10
MEAN_LIMIT = 10
hist_x = []
hist_y = []
fps = []
diff_hist = []
mean_hist = []
max_means = []
curr_bw = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
LOCK_MAX = 20
tap_lock = LOCK_MAX
while True:
    # For calculating FPS
    start_time = time.time()

    # Update prev and curr frames
    prev = curr_bw
    ret, curr = cap.read()
    curr_bw = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate difference between frames
    diff = curr_bw - prev
    diff[diff > 49] = 255

    # Refine difference matrix to isolate finger movement
    kernel = np.ones((8,5))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((9,9))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)

    # Store the order of magnitude for difference matrix
    diff_sum = np.sum(diff)
    diff_mag = len(str(diff_sum))
    diff_hist.append(diff_mag)
    if (len(diff_hist) > DIFF_LIMIT):
        diff_hist.pop(0)

    # Store the mean of the difference history
    curr_mean = sum(diff_hist)/len(diff_hist)
    mean_hist.append(curr_mean)
    if (len(mean_hist) > MEAN_LIMIT):
        mean_hist.pop(0)

    # Check if a tap action is performed
    tap_lock -= 1
    max_mean = max(mean_hist)
    max_means.append(max_mean)
    if (len(max_means) >= MEAN_LIMIT):
        max_means.pop(0)

    #print(max_mean)
    #print(max_means[-1])
    #print(max_means)
    if (len(mean_hist) == 10):
        #if ((max_means.count(max_mean) >= 4 and (max_means[0] < max_means[1] and max_means[-1] < max_means[-2])) or max_means.count(max_mean) == MEAN_LIMIT):
        if (max_means.count(max(max_means)) >= 4 and max(max_means) >= 5.6 and max(max_means) <= 5.9):
        #if (max_means.count(max(max_means)) >= 4 and (max_means[0] < max_means[1] or max_means[-1] < max_means[2])):
            if (tap_lock <= 0):
                pag.click()
                print("tapped??\n")
                tap_lock = LOCK_MAX
    
    hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)

    #lower_hand = np.array([41,57,83])
    #upper_hand = np.array([81,93,161])
    
    # Isolate the hand via HSV region selection
    lower_hand = np.array([59,0,120])
    upper_hand = np.array([137,110,255])
    mask_hand = cv2.inRange(curr, lower_hand, upper_hand)

    # Isolate the keyboard via HSV region selection
    lower_keyboard = np.array([79,0,0])
    upper_keyboard = np.array([167,229,65])
    res = cv2.inRange(hsv, lower_keyboard, upper_keyboard)
    kernel = np.ones((16,16))
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((20,20))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

    # Find the convex hull around the keyboard
    keyboard_mask, dims = largestConvexHull(res, curr)
    key_x, key_y, key_w, key_h = dims

    # Only keep pixels where both hand and keyboard are present
    # (i.e. only keep finger pixels within the keyboard)
    both = 255*((mask_hand/255) * (keyboard_mask/255))

    # Extract the tip of the finger
    x, y = getLargestCentroid(np.uint8(both))


    if (x != -1 and y != -1):
        # Adjust X so it is measured relative to a cropped horizontal keyboard area
        min_x, max_x = [250, 650]
        rel_x = (x - min_x) / (max_x - min_x)
        if rel_x > 1:
            rel_x = 1
        if rel_x < 0:
            rel_x = 0
        
        # Adjust Y so it is measured relative to a cropped vertical keyboard area
        min_y, max_y = [310, 370]
        rel_y = (y - min_y) / (max_y - min_y)
        if rel_y > 1:
            rel_y = 1
        if rel_y < 0:
            rel_y = 0

        # Calculate absolute screen coordinates
        absolute_x = pag.size()[0] - pag.size()[0] * rel_x
        absolute_y = pag.size()[1] * rel_y

        # Update the running histories for x and y coords
        # Used for smoothing motion (at the cost of increased latency)
        hist_x.append(absolute_x)
        hist_y.append(absolute_y)
        if (len(hist_x) > HIST_LIMIT):
            hist_x.pop(0)
            hist_y.pop(0)

        # Move the mouse
        pag.moveTo(sum(hist_x)/len(hist_x), sum(hist_y)/len(hist_y))

    fps.append(1.0 / (time.time() - start_time))
    #print(sum(fps) / len(fps))
