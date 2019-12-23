import cv2
import numpy as np
from edge import Edge
from filter_noise import FilterNoise
from utils import *
import sys


class LineDetector:
    def __init__(self, func, threshold):
        self.ht_rho = 1
        self.ht_theta = np.pi / 180
        self.ht_threshold = threshold
        
        if func == "houghlines":
            self.get_lines = self.houghlines

    def houghlines(self, img):
        return cv2.HoughLines(img, self.ht_rho,  self.ht_theta,  self.ht_threshold)



if __name__ == '__main__':
    images = ['image_raw_1.png', 'image_raw_2.png', 'image_raw_3.png', 'image_raw_0.png']
    
    filterObj     = FilterNoise("blur", 3)
    edgesObj      = Edge("sobel")
    lineDetectorObj = LineDetector("houghlines")

    for file in images:
        img = cv2.imread("./tests/" + str(file))

        img_H, img_W, _ = img.shape
        
        # image pre processing step
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered = filterObj.get_img(gray)
        edges = edgesObj.get_edges(filtered)
        
        # hough lines detection step
        hlines = lineDetectorObj.get_lines(edges)

        lines = set_plt_msgs_Lines(hlines)
        img_lines = draw_lines(img, lines, RED)
        
        rt = get_best_lines(lines, 3)
        for rho, theta in rt:
            pt1, pt2 = get_points( rho, theta )
            cv2.line(img_lines, pt1, pt2, BLUE, thickness=2)
            print(rho, theta)

        cv2.imshow("houghlines", img_lines)
        cv2.waitKey(0)
        print("\n\n\n")