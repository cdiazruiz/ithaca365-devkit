import cv2

def apply_rectify(instrinsic, distCoeff, R, P, size, img):
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix=instrinsic,
        distCoeffs=distCoeff,
        R=R,
        newCameraMatrix=P,
        size=size,
        m1type=cv2.CV_32FC1)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
