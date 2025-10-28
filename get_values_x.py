# This library contains the functions for feature extraction written by Xander, all functions take a image as input and output 
#   one value so they can be used in the functions from test_random.py and add_to_dataframe.py

# Library imports
import numpy as np
import cv2

def get_mean_hue(img, exclude_black=True):
    """This function returns the mean hue value of the input image, with te possibility of excluding black pixels"""
    #   Convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #   If black should be excluded only process non-black pixels 
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        hue_values = hsv[:,:,0][mask > 0]
    else:
        hue_values = hsv[:,:,0].ravel()
    
    #   Return None if no non-black pixels are found
    if hue_values.size == 0:
        return None

    #   Get mean of hue values
    mean_hue = np.mean(hue_values)

    return mean_hue

def get_mean_sat(img, exclude_black=True):
    """This function returns the mean saturation value of the input image, with te possibility of excluding black pixels"""
    #   Convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #   If black should be excluded only process non-black pixels 
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        sat_values = hsv[:,:,1][mask > 0]
    else:
        sat_values = hsv[:,:,1].ravel()
    
    #   Return None if no non-black pixels are found
    if sat_values.size == 0:
        return None  # No valid hue values found

    #   Get mean of saturation values
    mean_sat = np.mean(sat_values)

    return mean_sat

def get_mean_val(img, exclude_black=True):
    """This function returns the mean value value of the input image, with te possibility of excluding black pixels"""
    #   Convert to hsv colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #   If black should be excluded only process non-black pixels 
    if exclude_black:
        mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        val_values = hsv[:,:,2][mask > 0]
    else:
        val_values = hsv[:,:,2].ravel()
    
    #   Return None if no non-black pixels are found
    if val_values.size == 0:
        return None  # No valid hue values found

    #   Get mean of value values
    mean_val = np.mean(val_values)

    return mean_val

def get_edge_percentage(img, ignore_black=True):
    """This function returns the percentage of pixels recognized by canny edge, with te possibility of excluding black pixels"""
    #   Convert to gray colorspace
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #   Detect edges using canny edge
    edges = cv2.Canny(gray, 100, 150)
    
    #   If black should be excluded only process non-black pixels 
    if ignore_black:
        mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    #   Count edge pixels
    edge_pixels = np.sum(edges > 0)
    #   Count total pixels
    total_pixels = np.sum(mask > 0) if ignore_black else img.shape[0] * img.shape[1]
    
    #   Avoid division by zero
    if total_pixels == 0:
        return 0.0 
    
    #   Calculate edge percentage
    edge_percentage = (edge_pixels / total_pixels) * 100

    return edge_percentage

def get_pizza_radius(img, testing=False):
    "This function returns the radius of the pizza in pixels"
    #   Convert to gray colorspace and blur image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (5, 5))

    # Detect circles using hough circles algorithm
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

    #   Calculate radius of biggest detected circle, if no circle is found return average of image heigth and width
    if circles is not None:
        #   Convert content of circles to integer
        circles = np.uint16(np.around(circles))
        
        #   Get coordinates and radius of closest circle
        x, y, r = circles[0][0]

        #   Show found circle on image if testing is enabled
        if testing:
            #   Copy input image
            output = img.copy()
            
            # Draw circle outline and center point
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)  
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3) 

            #   Show output image
            cv2.imshow("output", output)
            cv2.waitKey(0)
        
        #   Return circle radius
        return r
    else:
        #   Get image height and width
        h, w = img.shape[:2]
        #   Return average of heigth and width
        return int((h+w)/2)

def get_small_circles(img, testing=False):
    """This function returns the amount of small circles on a image"""
    #   Convert to gray colorspace and blur image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (5, 5))

    #   Get radius of pizza
    r_pizza = get_pizza_radius(img)

    #   Find small circles on images
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(r_pizza/8), #   minimum distance between circles is 1/8 of pizza radius
        param1=60,
        param2=23,
        minRadius=int(r_pizza/12),  #   minimum radius of circles is 1/12 of pizza radius
        maxRadius=int(r_pizza/6)    #   maximum radius of circles is 1/6 of pizza radius
    )

    #   Get amount of found circles, if none are found return 0
    if circles is not None:
        #   Convert content of circles to integer
        circles = np.uint16(np.around(circles))

        #   Show found circle on image if testing is enabled
        if testing:
            #   Copy input image
            output = img.copy()

            #   Loop over found circles
            for i in circles[0,:]:
                #   Get circle coordinate and radius
                x = i[0]
                y = i[1]
                r = i[2]
                
                #   Draw circle outline and centerpoint
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            #   Show output image
            cv2.imshow("output", output)
            cv2.waitKey(0)
        
        #   Return amount of found circles
        return len(circles[0,:])
    else:
        return 0
    
def get_med_circles(img, testing=False):
    """This function returns the amount of medium circles on a image"""
    #   Convert to gray colorspace and blur image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (5, 5))

    #   Get radius of pizza
    r_pizza = get_pizza_radius(img)

    #   Find medium circles on images
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(r_pizza/4), #   minimum distance between circles is 1/4 of pizza radius
        param1=100,
        param2=25,
        minRadius=int(r_pizza/6),   #   minimum radius of circles is 1/6 of pizza radius
        maxRadius=int(r_pizza/3)    #   maximum radius of circles is 1/3 of pizza radius
    )

    #   Get amount of found circles, if none are found return 0
    if circles is not None:
        #   Convert content of circles to integer
        circles = np.uint16(np.around(circles))

        #   Show found circle on image if testing is enabled
        if testing:
            #   Copy input image
            output = img.copy()

            #   Loop over found circles
            for i in circles[0,:]:
                #   Get circle coordinate and radius
                x = i[0]
                y = i[1]
                r = i[2]
                
                #   Draw circle outline and centerpoint
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            #   Show output image
            cv2.imshow("output", output)
            cv2.waitKey(0)
        
        #   Return amount of found circles
        return len(circles[0,:])
    else:
        return 0
    
def get_blobcount_s(img):
    """This function returns the amount of small blobs found on the input image"""
    #   Convert to gray colorspace
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #   Create and adjust blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 50     #   Minimum area is 50
    params.maxArea = 100    #   Maxumum area is 100
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True   #   Filter by inertia

    #   Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    #   Detect keypoints
    keypoints = detector.detect(img_gray)
    
    #   Return amount of found keypoints
    return len(keypoints)

def get_blobcount_m(img):
    """This function returns the amount of medium blobs found on the input image"""
    #   Convert to gray colorspace
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #   Create and adjust blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 100    #   Minimum area is 100
    params.maxArea = 400    #   Maxumum area is 400
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True   #   Filter by inertia

    #   Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    #   Detect keypoints
    keypoints = detector.detect(img_gray)
    
    #   Return amount of found keypoints
    return len(keypoints)

def get_blobcount_l(img):
    """This function returns the amount of large blobs found on the input image"""
    #   Convert to gray colorspace
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #   Create and adjust blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 400    #   Minimum area is 400
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = True   #   Filter by inertia

    #   Create blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    #   Detect keypoints
    keypoints = detector.detect(img_gray)
    
    #   Return amount of found keypoints
    return len(keypoints)

def get_cc_n(img):
    """This function returns the amount of connected contours found in the input image"""
    #   Convert to gray colorspace and blur image
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    #   Set threshold
    threshold = cv2.threshold(img_blur, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #   Get connected components analysis
    analysis = cv2.connectedComponentsWithStats(threshold,
                                            4,
                                            cv2.CV_32S)
    
    #   Extract amount of connected components
    (n_connected_components, label_ids, values, centroid) = analysis

    #   Return amount of connected components
    return n_connected_components

def get_cc_mean(img):
    """This function returns the mean reaction value of connected contours found in the input image"""
    #   Convert to gray colorspace and blur image
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    #   Set threshold
    threshold = cv2.threshold(img_blur, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #   Get connected components analysis
    analysis = cv2.connectedComponentsWithStats(threshold,
                                            4,
                                            cv2.CV_32S)
    
    #   Extract connected components reaction
    (n_connected_components, label_ids, values, centroid) = analysis

    #   Get mean reaction value
    mean_connected_components = values.mean()

    #   Return mean reaction value
    return mean_connected_components

def get_cc_mean_area(img):
    """This function returns the mean area of connected contours found in the input image"""
    #   Convert to gray colorspace and blur image
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    #   Set threshold
    threshold = cv2.threshold(img_blur, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    #   Get connected components analysis
    analysis = cv2.connectedComponentsWithStats(threshold,
                                            4,
                                            cv2.CV_32S)
    
    #   Extract connected components areas
    (n_connected_components, label_ids, values, centroid) = analysis
    areas = values[1:, cv2.CC_STAT_AREA]

    #   Return mean area
    return areas.mean()

def get_mean_des_sift(img):
    """This function returns the mean grayscale value of the pixels around sift keypoints"""
    #   Convert to gray colorspace
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #   Create sift
    sift = cv2.SIFT_create()

    #   Detect keypoints
    kp = sift.detect(gray, None)

    #   Filter keypoints for minimum response
    kp_filtered = [i for i in kp if i.response >= 0.05]

    #   Compurte descriptor for filtered keypoints
    kp_filtered, des = sift.compute(gray, kp_filtered)

    #   Return mean descriptor value, if no descriptors are found return 0
    try:
        return des.mean()
    except:
        return 0

def get_n_sift(img):
    """This function returns the amount of sift keypoints found"""
    #   Convert to gray colorspace
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #   Create sift
    sift = cv2.SIFT_create()

    #   Detect keypoints
    kp = sift.detect(gray, None)

    #   Filter keypoints for minimum response
    kp_filtered = [i for i in kp if i.response >= 0.05]

    #   Return amount of found keypoints
    return len(kp_filtered)
