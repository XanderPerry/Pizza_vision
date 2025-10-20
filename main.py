import cv2

import pizza_cutter
import test_random
import get_values_x


while True:
    imgs = test_random.get_random_images()

    cv2.imshow("original che", imgs["che"])

    mean_val = test_random.apply_function(get_values_x.get_mean_val, imgs)

    test_random.imgs_print_results(results_list=[mean_val], labels_list=["mean val"])

    key = cv2.waitKey(0)
    if key& 0xFF == 27:
        print("Exiting...")
        break
    else:
        cv2.destroyAllWindows()
        print("New images.")

cv2.destroyAllWindows()