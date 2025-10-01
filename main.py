import cv2

import pizza_cutter
import test_random
import whitebalance

while True:
    imgs = test_random.get_random_images()

    cv2.imshow("original che", imgs["che"])

    imgs_cutout = test_random.apply_function(pizza_cutter.cut_pizza, imgs)

    test_random.imgs_compare_visual(imgs_a=imgs, imgs_b=imgs_cutout)

    key = cv2.waitKey(0)
    if key& 0xFF == 27:
        print("Exiting...")
        break
    else:
        cv2.destroyAllWindows()
        print("New images.")

cv2.destroyAllWindows()