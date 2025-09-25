import cv2
import glob
import shutil

KINDS = ["che", "fun", "haw", "mar", "moz", "sal"]

for kind in KINDS:
    for_test = True
    i = 0
    DIRECTORY = "C:/Users/Xande/EE cursusmateriaal/3A/Beeldherkenning/PizzaVisionRepo/Pizza_vision/"

    for filename_old in glob.glob("data/" + kind + "/train/**/*.jpg", recursive=True):
        filename_new = DIRECTORY + "data_split/" + kind + "/train/" + filename_old[filename_old.find("\\")+1:filename_old.find("\\")+14] + str(i).zfill(4)
        i+=1
        
        # print(filename_new)
        shutil.copy2((DIRECTORY + filename_old), filename_new)


    j = 0
    for filename_old in glob.glob("data/" + kind + "/test/**/*.jpg", recursive=True):
        if for_test:
            filename_new = DIRECTORY + "data_split/" + kind + "/test/" + filename_old[filename_old.find("\\")+1:filename_old.find("\\")+14] + str(j).zfill(4) + ".jpg"
            for_test = False
        else:
            filename_new = DIRECTORY + "data_split/" + kind + "/validation/" + filename_old[filename_old.find("\\")+1:filename_old.find("\\")+14] + str(j).zfill(4) + ".jpg"
            for_test = True
            j+=1


        # print(filename_new)
        shutil.copy2((DIRECTORY + filename_old), filename_new)

