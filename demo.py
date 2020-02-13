from MST import MBDMSTree
import cv2

if __name__ == "__main__":

    img = cv2.imread('0027.png')
    mbdst = MBDMSTree(img)
    for i in range(mbdst.img_width):
        mbdst.is_seed[i] = True
        mbdst.is_seed[(mbdst.img_height-1) * mbdst.img_width + i] = True
    for i in range(mbdst.img_height):
        mbdst.is_seed[i * mbdst.img_width] = True
        mbdst.is_seed[i * mbdst.img_width + mbdst.img_width - 1] = True
    salmap = mbdst.compute_MBD()
    cv2.imwrite('salmap.png', salmap)