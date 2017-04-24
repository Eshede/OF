#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from math import sqrt


h = 0
w = 0
circles = []
size = (30,70)
circle_amount = 8
min_speed = 5
mi = 50
friction = 0.95


class Circle(object):
    def __init__(self, center, radius, color):
        self.cen = center
        self.rad = radius
        self.col = color
        self.elasticity = -1 - (radius - size[0])/float(size[1]-size[0])
        self.speed = np.array([0.,0.])

    def moving(self):
        if np.any(self.speed.astype(int)):
            new_cen = self.cen+self.speed.astype(int)
            if not (self.rad < new_cen.item(0) < w - self.rad):
                self.speed[0] /= self.elasticity
            if not (self.rad < new_cen.item(1) < h - self.rad):
                self.speed[1] /= self.elasticity
            self.cen += self.speed.astype(int)
            self.speed *= friction


def next_frame(img, flow, step=5):
    for i in circles:
        vectors = []
        for x in np.arange(-i.rad, i.rad, step):
            t = int(sqrt(i.rad**2 - x**2))
            for y in np.arange(-t, t, step):
                s = flow[i.cen[1]+x,i.cen[0]+y].T
                if np.linalg.norm(s) > min_speed:
                    vectors.append(np.array(s))
        if len(vectors)> mi:
            i.speed = np.mean(vectors, axis=0)
        i.moving()
        cv2.circle(img, tuple(i.cen), i.rad, i.col, -1)
    return img


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    ret, prev = cam.read()
    h, w = prev.shape[:2]
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i in range(circle_amount):
        circles.append(Circle(np.array([np.random.randint(size[1],h-size[1]), np.random.randint(size[1],h-size[1])]),
            np.random.randint(*size),
            (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))))
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prevgray = gray

        cv2.imshow('flow', next_frame(img, flow, 5))
        ch = cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()
