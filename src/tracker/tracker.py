import cv2
import numpy as np

from src.parser import ParserRAW

class Tracker:
    def __init__(self, bgr=False, mode=cv2.MOTION_AFFINE):
        self.warp_bgr = bgr
        self.warp_mode = mode
        self.warp_flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50,  1e-10)

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            self.warp_matrix = np.eye(3, 3, dtype=np.float32)
            self.warp_function = lambda i, j, k: cv2.warpPerspective(i, j, k, flags=self.warp_flags)
        else:
            self.warp_matrix = np.eye(2, 3, dtype=np.float32)
            self.warp_function = lambda i, j, k: cv2.warpAffine(i, j, k, flags=self.warp_flags)

        self.ref = None

    def find_edge(self, channel):
        grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    def process(self, frame):
        if self.ref is None:
            ref = frame.copy()
            if self.warp_bgr:
                ref = self.process_BGR(ref)

            ref = cv2.cvtColor(ref, cv2.COLOR_RGB2YUV)
            ref, _, _ = cv2.split(ref)
            self.ref = self.find_edge(ref)

        cur = frame.copy()
        if self.warp_bgr:
            cur = self.process_BGR(cur)
        cur = cv2.cvtColor(cur, cv2.COLOR_RGB2YUV)
        cur, _, _ = cv2.split(cur)
        cur = self.find_edge(cur)
        h, w, _ = frame.shape

        (cc, warp_matrix) = cv2.findTransformECC(self.ref, cur, self.warp_matrix, self.warp_mode, self.criteria)

        b = self.warp_function(frame[:, :, 0], warp_matrix, (w, h))
        g = self.warp_function(frame[:, :, 1], warp_matrix, (w, h))
        r = self.warp_function(frame[:, :, 2], warp_matrix, (w, h))

        cur = cv2.merge((b, g, r))

        return cur

    def process_BGR(self, frame):
        ret = frame.copy()
        sobel = [self.find_edge(frame[:, :, i]) for i in range(0, 3)]
        h, w, _ = frame.shape

        for i in range(0, 2):
            (cc, warp_matrix) = cv2.findTransformECC(sobel[2], sobel[i], self.warp_matrix, self.warp_mode, self.criteria)
            ret[:, :, i] = self.warp_function(frame[:, :, i], warp_matrix, (w, h))

        return ret

if __name__ == "__main__":
    """
    img = ParserRAW().read("../../images/Sony_A7-MK3/planets/jupiter/DSC00181.ARW")
    ret = Tracker().process_BGR(img)
    cv2.imshow("raw", img)
    cv2.imshow("ret", ret)
    cv2.waitKey()
    """

    T = Tracker(bgr=False, mode=cv2.MOTION_HOMOGRAPHY)
    B = cv2.VideoCapture("../../images/Unknown/planets/jupiter/blue-500.avi")
    G = cv2.VideoCapture("../../images/Unknown/planets/jupiter/green-500.avi")
    R = cv2.VideoCapture("../../images/Unknown/planets/jupiter/red-500.avi")

    vid = []

    while B.isOpened() and G.isOpened() and R.isOpened():
        _, b = B.read()
        _, g = G.read()
        _, r = R.read()

        if b is None or g is None or r is None:
            break

        img = cv2.merge([b[:, :, 0], g[:, :, 1], r[:, :, 2]])
        img = T.process(img)
        cv2.imshow("img", img)
        cv2.waitKey(1)