import cv2
import numpy as np
import sys

class FilterNoise:
    def __init__(self, func, kernel_size):
        self.kernel_size = kernel_size
        
        if func == "median":
            self.get_img = self.median
        elif func == "blur":
            self.get_img = self.blur
        elif func == "bilateral":
            self.get_img = self.bilateral
        elif func == "gaussian":
            self.get_img = self.gaussian
        else:
            self.get_img = self.none

    def median(self, img):
        return cv2.medianBlur(img, self.kernel_size)

    def blur(self, img):
        return cv2.blur(img, (self.kernel_size, self.kernel_size))

    def bilateral(self, img):
        return cv2.bilateralFilter(img, self.kernel_size, 75, 75) # geralmente kernel_size eh < 10

    def gaussian(self, img):
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0) 
    
    def none(self, img):
        return img

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    print ('test filter')

    images = ['image_raw_1.png', 'image_raw_2.png', 'image_raw_3.png', 'image_raw_0.png']

    for file in images:
        img = cv2.imread("./tests/" + str(file))
        kernel_size = (int)(sys.argv[1])

        img_H, img_W, _ = img.shape
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        F1 = FilterNoise("median", kernel_size)
        F2 = FilterNoise("blur", kernel_size)
        F3 = FilterNoise("bilateral", kernel_size)
        F4 = FilterNoise("gaussian", kernel_size)

        R1 = F1.get_img(gray)
        R2 = F2.get_img(gray)
        R3 = F3.get_img(gray)
        R4 = F4.get_img(gray)

        cv2.imshow("median", R1)
        cv2.waitKey(0)
        cv2.destroyWindow("median")
        
        cv2.imshow("blur", R2)
        cv2.waitKey(0)
        cv2.destroyWindow("blur")

        cv2.imshow("bilateral", R3)
        cv2.waitKey(0)
        cv2.destroyWindow("bilateral")

        cv2.imshow("gaussian", R4)
        cv2.waitKey(0)
        cv2.destroyWindow("gaussian")