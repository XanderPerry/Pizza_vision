import cv2

import pizza_cutter
import test_random

while True:
    imgs = test_random.get_random_images()

    # imgs_circle = test_random.apply_function(pizza_cutter.cutout_circle, imgs)

    imgs_colormask = test_random.apply_function(pizza_cutter.crop_image, imgs)

    # imgs_combined = test_random.apply_function(pizza_cutter.cutout_circle, imgs_colormask)

    # test_random.imgs_compare_visual(imgs_a=imgs, imgs_b=imgs_circle, label_a=" original", label_b=" cutout circle")

    # test_random.imgs_compare_visual(imgs_a=imgs, imgs_b=imgs_colormask, label_a=" original", label_b=" cropped")

    # test_random.imgs_compare_visual(imgs_a=imgs, imgs_b=imgs_combined, label_a=" original", label_b=" combined")

    cv2.imshow("orig che", imgs["che"])

    # test_random.imgs_compare_visual_list(imgs_list=[imgs, imgs_circle, imgs_colormask, imgs_combined], labels_list=[" original", " circle crop", " colormask crop", " combined crop"])

    test_random.imgs_compare_visual_list(imgs_list=[imgs, imgs_colormask], labels_list=[" original", " colormask crop"])

    key = cv2.waitKey(0)
    if key& 0xFF == 27:
        print("Exiting...")
        break
    else:
        cv2.destroyAllWindows()
        print("New images.")

cv2.destroyAllWindows()