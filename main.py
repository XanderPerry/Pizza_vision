import cv2

import pizza_cutter
import test_random
import get_values_x
import get_values_h


while True:
    imgs = test_random.get_random_images()

    cv2.imshow("original che", imgs["che"])

    # get_LBP = test_random.apply_function(get_values_h.get_fourth_element_LBP, imgs)

    # test color percentages extraction:
    get_red = test_random.apply_function(get_values_h.get_red_percentages, imgs)
    # get_green = test_random.apply_function(get_values_h.get_green_percentages, imgs)
    # get_yellow = test_random.apply_function(get_values_h.get_yellow_percentages, imgs)


    test_random.imgs_print_results(results_list=[get_red], labels_list=["get_red"])

    key = cv2.waitKey(0)
    if key& 0xFF == 27:
        print("Exiting...")
        break
    else:
        cv2.destroyAllWindows()
        print("New images.")

cv2.destroyAllWindows()