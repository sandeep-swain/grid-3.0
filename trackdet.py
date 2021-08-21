import cv2
import numpy as np
from collections import defaultdict

s1 = 0
s2 = 0


def line_intersection(line1, line2, th, pts):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    if (int(x) > s1[0] or int(y) > s1[1]) & (int(x) < s2[0] or int(y) > s1[1]):
        temp = []
        temp.append(x)
        temp.append(y)
        # print(x, y)
        # th = cv2.circle(th, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
        pts.append(temp)


def cell(corners , th):
    x = []
    y = []
    for i in range(86):
        x.append(corners[i][0])
        y.append(corners[i][1])
    x = set(x)
    x = list(x)
    x.sort()
    xcentr = []
    for i in range(len(x) - 1):
        if (x[i + 1] - x[i]) > 20:
            xcentr.append((x[i + 1] + x[i]) / 2)
    #print(xcentr)
    y = set(y)
    y = list(y)
    y.sort()
    ycentr = []
    for i in range(len(y) - 1):
        if (y[i + 1] - y[i]) > 20:
            ycentr.append((y[i + 1] + y[i]) / 2)
    #print(ycentr)
    centrcod = []
    for j in range(len(ycentr)):
        for i in range(len(xcentr)):
            if (int(xcentr[i]) > s1[0] or int(ycentr[j]) > s1[1]) & (int(xcentr[i]) < s2[0] or int(ycentr[j]) > s1[1]):
                print(xcentr[i],ycentr[j])
                th = cv2.circle(th, (int(xcentr[i]), int(ycentr[j])), radius=3, color=(0, 0, 255), thickness=-1)



def expt(lines, th):
    v = np.array([[0, 0, 0, 0]])
    h = np.array([[0, 0, 0, 0]])
    for line in lines:
        if abs(line[0][0] - line[0][2]) == 0:
            v = np.append(v, line, axis=0)

        if abs(line[0][1] - line[0][3]) == 0:
            h = np.append(h, line, axis=0)
    I = v.shape[0]
    J = h.shape[0]
    pts = []
    for i in range(1, I):
        for j in range(1, J):

            line_intersection(([v[i][0], v[i][1]], [v[i][2], v[i][3]]), ([h[j][0], h[j][1]], [h[j][2], h[j][3]]), th,
                              pts)

    # print(pts)
    print(len(pts))
    pts = sorted(pts, key=lambda x: x[1])
    L = np.array(pts)
    # print(L)
    print("xxxxxxxxxxxxx")
    # print(np.unique(xy, axis=0))
    # L = np.unique(xy, axis=0)
    L = np.float32(L)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 520, 0.000000001)
    ret, label, center = cv2.kmeans(L, 86, None, criteria, 100, cv2.KMEANS_PP_CENTERS)
    center = sorted(center, key=lambda x: x[1])
    center = sorted(center, key=lambda x: x[0])
    center = np.array(center, dtype=int)
    cell(center, th)
    # for i in range(len(center)):
    #     th = cv2.circle(th, (int(center[i][0]), int(center[i][1])), radius=3, color=(250, 0, 255), thickness=-1)


def Hough(img, img2):
    rho, theta, thresh = 1, np.pi / 180, 52
    # lines = cv2.HoughLines(img, rho, theta, thresh)
    lines = cv2.HoughLinesP(img, 1, theta, thresh, minLineLength=10, maxLineGap=10)
    # segmented = segment_by_angle_kmeans(lines)
    # intersections = segmented_intersections(segmented)
    expt(lines, img2)
    # print(lines)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)


def getcontours(vdo, th1):
    contours, hierarchy = cv2.findContours(vdo, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 60000:
            # cv2.drawContours(th1, contours, -1, (0, 255, 0), 3)

            peri = cv2.arcLength(contour, True)

            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            global s1
            global s2
            s1 = approx[4][0]
            s2 = approx[1][0]

            objcor = len(approx)
            # print(objcor)
            # print(area)
            # print(s1[1])
            th1 = cv2.rectangle(th1, (0, 0), (s1[0] - 4, s1[1] - 4), (0, 0, 0), -1)
            th1 = cv2.rectangle(th1, (692, 0), (s2[0] + 4, s2[1] - 4), (0, 0, 0), -1)
            x, y, w, h = cv2.boundingRect(approx)


frame = cv2.imread("arena.png")
blur = cv2.GaussianBlur(frame, (5, 5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 0, 10])
upper_red = np.array([0, 0, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
th1 = cv2.bitwise_and(frame, frame, mask=mask)
th1 = cv2.erode(th1, None, iterations=2)
th1 = cv2.dilate(th1, None, iterations=2)
getcontours(mask, th1)
gray = cv2.cvtColor(th1, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 150, apertureSize=3)
Hough(edges, th1)

cv2.imshow('th1', th1)
cv2.imshow("frame", frame)
cv2.imshow("edges", edges)

cv2.waitKey(0)

cv2.destroyAllWindows()
