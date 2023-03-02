from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import random
import cv2
import time


class OperateDetect(object):

    def __init__(self):
        self.weight = "./shape_predictor_68_face_landmarks.dat"
        self.EYE_AR_THRESH = 0.25  # 眼睛对应的长宽比低于该阈值算一次眨眼
        self.CONSEC_FRAMES = 5  # 这个值被设置为 3，表明眼睛长宽比小于3时，接着三个连续的帧一定发生眨眼动作
        self.COUNTER = 0  # 眼图长宽比小于EYE_AR_THRESH的连续帧的总数
        self.TOTAL = 0  # 脚本运行时发生的眨眼的总次数
        self.MOUTH_AR_THRESH = 0.8  # 张嘴的阈值
        self.Nod_threshold = 0.03  # 点头的阈值
        self.shake_threshold = 0.03  # 摇头的阈值
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.weight)
        self.mStart, self.mEnd = 49, 68  # 嘴部对应的索引
        self.nStart, self.nEnd = 32, 36  # 鼻子对应的索引
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # 左眼对应的的索引
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]  # 右眼对应的索引
        self.compare_point = [0, 0]  # 刚开始的时候设置鼻子的中点在【0， 0】

    def eye_aspect_ratio(self, eye):
        '''
        A，B是计算两组垂直眼睛标志之间的距离，而C是计算水平眼睛标志之间的距离
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        :param eye:两个眼睛在图片中的像素坐标
        :return:返回眼睛的长宽比
        '''

        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)  # 将分子和分母相结合，得出最终的眼睛纵横比。然后将眼图长宽比返回给调用函数
        return ear

    def mouth_aspect_ratio(self, mouth):
        # 计算两组垂直方向的欧式距离
        A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
        B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
        # 计算水平方向的距离
        C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def center_point(self, nose):
        # 求nose数组中所有元素的均值
        return nose.mean(axis=0)

    def nod_aspect_ratio(self, size, pre_point, now_point):
        return abs(float((pre_point[1] - now_point[1]) / (size[0] / 2)))

    def shake_aspect_ratio(self, size, pre_point, now_point):
        return abs(float((pre_point[0] - now_point[0]) / (size[1] / 2)))

    def blinks_detect(self, shape):
        # 取左右眼对应的索引
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        # 计算左右眼纵横比
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # 计算左右眼的凸包，并可视化
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        return ear, leftEyeHull, rightEyeHull

    def mouth_detect(self, shape):
        mouth = shape[self.mStart:self.mEnd]
        # 计算嘴的纵横比
        mouthMAR = self.mouth_aspect_ratio(mouth)
        mar = mouthMAR
        # 计算嘴巴的凸包
        mouthHull = cv2.convexHull(mouth)
        return mar, mouthHull

    def nod_shark(self, size, shape):
        # 取鼻子的索引
        nose = shape[self.nStart:self.nEnd]
        nose_center = self.center_point(nose)
        nod_value, shake_value = 0, 0
        # 计算鼻子的凸包
        noseHull = cv2.convexHull(nose)
        if self.compare_point[0] != 0:
            nod_value = self.nod_aspect_ratio(size, nose_center, self.compare_point)
            shake_value = self.shake_aspect_ratio(size, nose_center, self.compare_point)
        self.compare_point = nose_center
        return nod_value, shake_value, noseHull

    # 判断脸是否在规定区域
    def in_area(self, key_points, circle_center, radius):
        result = True
        left_eyebrow = key_points[18:22]
        right_eyebrow = key_points[18:22]
        mouth = key_points[49:68]
        left_eyebrow_center = self.center_point(left_eyebrow)
        right_eyebrow_center = self.center_point(right_eyebrow)
        mouth_center = self.center_point(mouth)
        # 首先判断三个点的是否在圆内
        if np.sqrt(np.sum((circle_center - left_eyebrow_center) ** 2)) - radius > 0:
            result = False
        if np.sqrt(np.sum((circle_center - right_eyebrow_center) ** 2)) - radius > 0:
            result = False
        if np.sqrt(np.sum((circle_center - mouth_center) ** 2)) - radius > 0:
            result = False
        if left_eyebrow_center[0] > circle_center[0] or left_eyebrow_center[1] > circle_center[1]:
            result = False
        if right_eyebrow_center[0] < circle_center[0] * 0.9 or left_eyebrow_center[1] > circle_center[1]:
            result = False
        if mouth_center[1] < circle_center[1] + radius / 2:
            result = False
        return result

    # 记录通过验证的项目
    def action_judgment(self, action_value):
        action_type = np.array([0, 0, 0, 0])
        ear, mar, nod_value, shake_value = action_value
        if ear < self.EYE_AR_THRESH:
            action_type[0] = 1
        if mar > self.MOUTH_AR_THRESH:
            action_type[1] = 1
        if nod_value > self.Nod_threshold:
            action_type[2] = 1
        if shake_value > self.shake_threshold:
            action_type[3] = 1
        return action_type

    def detect(self, frame):
        img = imutils.resize(frame, width=640)
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 返回人脸矩形框两个点坐标：左上角(x1,y1)、右下角(x2,y2)
        rects = self.detector(gray, 0)
        size = frame.shape
        if len(rects) != 0:
            for rect in rects:  # 遍历帧中的每个面，然后对其中的每个面应用面部标志检测
                # shape确定面部区域的面部标志，接着将这些（x，y）坐标转换成NumPy阵列
                # shape为人脸68特征点检测器返回的68个特征点位置
                shape = self.predictor(img, rect)
                # shape转换成numpy序列
                shape = face_utils.shape_to_np(shape)
                # 判断脸是否在规定区域
                in_circle = self.in_area(shape, np.array([320, 180]), 90)
                #检测结果
                ear, leftEyeHull, rightEyeHull = self.blinks_detect(shape)
                mar, mouthHull = self.mouth_detect(shape)
                nod_value, shake_value, noseHull = self.nod_shark(size, shape)
                act_type = self.action_judgment((ear, mar, nod_value, shake_value))
                return in_circle, act_type, leftEyeHull, rightEyeHull, mouthHull, noseHull

def resultPrint(index):
    if index == 0:
        print("眨眼测试通过！")
        return
    if index == 1:
        print("张嘴测试通过！")
        return
    if index == 2:
        print("点头测试通过！")
        return
    if index == 3:
        print("摇头测试通过！")
        return


def main():
    vs = VideoStream(src=0).start()
    live_detect = OperateDetect()
    # 每次运行的时候做三种方式的检测， 顺序打乱
    detect_type = [0, 1, 2, 3]
    random.shuffle(detect_type)
    # 取从左一元素到倒数第一元素
    detect_type = detect_type[0:-1]
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    size = frame.shape
    activate_judge = np.array([0, 0, 0, 0])
    first_frame_type = True
    # 持续刷新
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=640)
        # 在屏幕中央画个圈
        cv2.circle(frame, (int(size[1] / 2), int(size[0] / 2)), int(min(size[0] / 2, size[1] / 2) * 0.5), (0, 255, 255), 3, cv2.LINE_AA)
        detect_result = live_detect.detect(frame)
        operate_step = True
        if len(detect_type) == 0:
            break
        if detect_result:
            if (len(detect_result)) == 6:
                is_align, act_info, leftEye, rightEye, mouth_point, nose_point = detect_result
                # 选取识别项目
                detect_num = detect_type[0]
                if first_frame_type:
                    if not is_align:
                        operate_step = False
                if operate_step:
                    if detect_num == 0:
                        cv2.putText(frame, "Please blink", (200, 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    if detect_num == 1:  # 张嘴操作
                        cv2.putText(frame, "Please open your mouth", (200, 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    if detect_num == 2:  # 点头
                        cv2.putText(frame, "Please nod", (200, 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    if detect_num == 3:  # 摇头操作
                        cv2.putText(frame, "Please shake your head", (200, 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    if is_align:
                        first_frame_type = False

                    if act_info[detect_num] == 1:
                        # 叠加检测到的次数
                        activate_judge = activate_judge + act_info
                        # 检测到3次以上动作认为通过
                        if activate_judge[detect_num] > 3:
                            cv2.putText(frame, "Liveness detection completed!", (30, 60),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                            resultPrint(detect_type[0])
                            # print("detect_type[0]", detect_type[0])
                            detect_type.remove(detect_type[0])
                            first_frame_type = True

                    else:
                        activate_judge = np.array([0, 0, 0, 0])
                    cv2.drawContours(frame, [leftEye], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEye], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [mouth_point], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [nose_point], -1, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Please align your face inside the circle", (30, 60),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if len(detect_type) == 0:
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
