import cv2
import numpy as np

from src.parser import ParserRAW

class Tracker:
    def __init__(self, mode=cv2.MOTION_AFFINE):
        self.warp_mode = mode
        self.warp_flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50,  1e-10)

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            self.warp_matrix = np.eye(3, 3, dtype=np.float32)
            self.warp_function = lambda i, j, k: cv2.warpPerspective(i, j, k, flags=self.warp_flags)
        else:
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            self.warp_function = lambda i, j, k: cv2.warpAffine(i, j, k, flags=self.warp_flags)

    def process(self, frame):
        def sobel_filter(channel):
            grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
            return cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

        ret = frame.copy()
        sobel = [sobel_filter(frame[:, :, i]) for i in range(0, 3)]
        h, w, _ = frame.shape

        for i in range(0, 2):
            (cc, warp_matrix) = cv2.findTransformECC(sobel[2], sobel[i], self.warp_matrix, self.warp_mode, self.criteria)
            ret[:, :, i] = self.warp_function(frame[:, :, i], warp_matrix, (w, h))

        return ret

if __name__ == "__main__":

    img = ParserRAW().read("../../images/Sony_A7-MK3/planets/jupiter/DSC00181.ARW")
    ret = Tracker().process(img)

    cv2.imshow("raw", img)
    cv2.imshow("ret", ret)
    cv2.waitKey()