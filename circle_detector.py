import numpy as np

import cv2
import glob

def calc_sobel(img_gray):
    # Apply Sobel operator
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    
    # Compute gradient magnitude
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
     
    # Convert to uint8
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    
    return gradient_magnitude

def calc_laplacian(img_gray):
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
     
    # Convert to uint8
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    
    return laplacian_abs

def calc_canny(img_gray):
    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
     
    # Apply Canny Edge Detector
    edges = cv2.Canny(blur, threshold1=50, threshold2=100)

    return edges

def hough_detect_circle(img_gray, img):
    img_gray = cv2.medianBlur(img_gray, 5)
    
    output = img.copy()
    
    # Detect circles
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,      
        param1=100,         
        param2=20,       
        minRadius=100,       
        maxRadius=300
    )
    
    # Draw only the first detected circle
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Circle outline
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point
    
    return output

for filename in glob.glob("image_capture/data/sal/train/**/*.jpg", recursive=True):
    # Import image and make copy in grayscale
    img = cv2.imread(filename)
    #cv2.imshow("original", img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate edge detection algorithms
    img_gray_sobel = calc_sobel(img_gray)
    img_gray_laplacian = calc_laplacian(img_gray)
    img_gray_canny= calc_canny(img_gray)
    
    # Calculate circle detection algorithms
    img_hough_circle = hough_detect_circle(img_gray, img)
    img_hough_circle_sobel = hough_detect_circle(img_gray_sobel, img)
    img_hough_circle_laplacian= hough_detect_circle(img_gray_laplacian, img)
    img_hough_circle_canny = hough_detect_circle(img_gray_canny, img)
    
    # Display results
    #cv2.imshow("Sobel Edge Detection", img_gray_sobel)
    #cv2.imshow("Laplacian Edge Detection", img_gray_laplacian)
    #cv2.imshow("Canny Edge Detection", img_gray_canny)
    
    cv2.imshow("Hough Detection - Original", img_hough_circle)
    cv2.imshow("Hough Detection - Sobel", img_hough_circle_sobel)
    cv2.imshow("Hough Detection - Laplacian", img_hough_circle_laplacian)
    cv2.imshow("Hough Detection - Canny", img_hough_circle_canny)
    
    
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        print("Manual interruption by user.")
        break
    
    cv2.destroyAllWindows()

cv2.destroyAllWindows()                           


     
    