import numpy as np
from skimage.feature import local_binary_pattern

import cv2

def get_mean_hue(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        hue_values = hsv[:,:,0][mask > 0]
    else:
        hue_values = hsv[:,:,0].ravel()
    
    if hue_values.size == 0:
        return None  # No valid hue values found

    mean_hue = np.mean(hue_values)
    return mean_hue

def get_mean_sat(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        sat_values = hsv[:,:,1][mask > 0]
    else:
        sat_values = hsv[:,:,1].ravel()
    
    if sat_values.size == 0:
        return None  # No valid hue values found

    mean_sat = np.mean(sat_values)
    return mean_sat

def get_mean_val(img, exclude_black=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        val_values = hsv[:,:,2][mask > 0]
    else:
        val_values = hsv[:,:,2].ravel()
    
    if val_values.size == 0:
        return None  # No valid hue values found

    mean_val = np.mean(val_values)
    return mean_val

def get_edge_percentage(img, ignore_black=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 100, 150)
    
    if ignore_black:
        mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    
    cv2.imshow("Canny Edges", edges)
    edge_pixels = np.sum(edges > 0)
    total_pixels = np.sum(mask > 0) if ignore_black else img.shape[0] * img.shape[1]
    
    if total_pixels == 0:
        return 0.0  # Avoid division by zero
    
    edge_percentage = (edge_pixels / total_pixels) * 100
    return edge_percentage

def get_pizza_radius(img, testing=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (5, 5))

    # Detect circles
    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=200,
        param1=100,
        param2=20,
        minRadius=100,
        maxRadius=300
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
    
        if testing:
            output = img.copy()
            
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Circle outline
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point

            cv2.imshow("output", output)
            cv2.waitKey(0)

        return r
    else:
        h, w = img.shape[:2]
        return int((h+w)/2)

def get_small_circles(img, testing=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (5, 5))

    r_pizza = get_pizza_radius(img)

    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(r_pizza/8),
        param1=60,
        param2=23,
        minRadius=int(r_pizza/12),
        maxRadius=int(r_pizza/6)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        if testing:
            output = img.copy()
            for i in circles[0,:]:
                x = i[0]
                y = i[1]
                r = i[2]
                
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Circle outline
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point

            cv2.imshow("output", output)
            cv2.waitKey(0)
        
        return len(circles[0,:])
    else:
        return 0
    
def get_med_circles(img, testing=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (5, 5))

    r_pizza = get_pizza_radius(img)

    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(r_pizza/4),
        param1=100,
        param2=25,
        minRadius=int(r_pizza/6),
        maxRadius=int(r_pizza/3)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        if testing:
            output = img.copy()
            for i in circles[0,:]:
                x = i[0]
                y = i[1]
                r = i[2]
                
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Circle outline
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point

            cv2.imshow("output", output)
            cv2.waitKey(0)
        
        return len(circles[0,:])
    else:
        return 0