import cv2
import numpy as np
import sys

class Edge:
    def __init__(self, func):
        self.canny_min = 100
        self.canny_max = 200
        
        if func == "canny":
            self.get_edges = self.canny
        elif func == "laplacian":
            self.get_edges = self.laplacian
        elif func == "sobel":
            self.get_edges = self.sobel

    def canny(self, img):
        return cv2.Canny(img, self.canny_min, self.canny_max)

    def laplacian(self, img):
        laplacian64f = cv2.Laplacian(img,cv2.CV_64F)
        abs_laplacian64f = np.absolute(laplacian64f)
        laplacian_8u = np.uint8(abs_laplacian64f)
        
        ret,thresh1 = cv2.threshold(laplacian_8u, 50, 255, cv2.THRESH_BINARY)
        return thresh1

    def sobel(self, img):
        sobel_img = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
        ret,thresh1 = cv2.threshold(sobel_img, 50, 255, cv2.THRESH_BINARY)
        return thresh1


def usage():
    print ('pls run:\npython edge.py <algorithm>\nwhere <algorithm> can be sobel, canny or laplacian')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print usage()
        exit(0)
    edge = Edge(sys.argv[1])

    img = cv2.imread('./tests/image_raw_0.png', 0)
    img = cv2.blur(img, (3,3))

    cv2.imshow("original", img)
    
    image_edges = edge.get_edges(img)
    cv2.imshow("edges", image_edges)
    cv2.waitKey(0)