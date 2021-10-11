#!/usr/bin/env python
import rospy
import torch
import threading
from queue import Queue
from yacs.config import CfgNode
from network.lcnn import LCNN
from network.dataset import Dataset
from std_msgs.msg import String
import PIL.Image as PILImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from lines2d_msgs.msg import lines2d
import cv2
import numpy as np
import os
import util.bezier as bez


class Node:
    maxsize = 10

    def __init__(self, cfg, sensor_file=''):
        # Params
        self.cameraMatrix = None
        self.distCoeffs = None
        self.newCameraMatrix = None
        self.lock = threading.Lock()
        self.image_buf = Queue(self.maxsize)
        self.br = CvBridge()
        self.publish_image = cfg.publish_image

        # Use GPU or CPU
        use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(cfg.gpu) if use_gpu else 'cpu')
        print('use_gpu: ', use_gpu)

        # Load model
        self.model = LCNN(cfg).to(self.device)
        model_filename = os.path.join(cfg.model_path, cfg.model_name)
        self.model.load_state_dict(torch.load(model_filename, map_location=self.device))
        self.model.eval()

        self.dataset = Dataset('', cfg, with_label=False, augment=False)

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(100)

        # Publishers
        self.pub_lines = rospy.Publisher('~lines2d', lines2d, queue_size=1000)
        if self.publish_image:
            self.pub_image = rospy.Publisher('~image_lines', Image, queue_size=1000)

        # Subscribers
        rospy.Subscriber('image', Image, self.callback)

        # camera parameters
        if sensor_file != '':
            fs = cv2.FileStorage(sensor_file, cv2.FileStorage_READ)
            ns = fs.getNode('projection_parameters')
            self.cameraMatrix = np.array([[ns.getNode('fx').real(), 0, ns.getNode('cx').real()],
                [0, ns.getNode('fy').real(), ns.getNode('cy').real()],
                [0, 0, 1]])
            ns = fs.getNode('distortion_parameters')
            self.distCoeffs = np.array([ns.getNode('k1').real(), ns.getNode('k2').real(), ns.getNode('p1').real(), ns.getNode('p2').real()])
            width = int(fs.getNode('image_width').real())
            height = int(fs.getNode('image_height').real())
            self.newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(self.cameraMatrix, self.distCoeffs, (width, height) , 0, (width, height))
            fs.release()

    def callback(self, image_msg):
        self.lock.acquire()
        if self.image_buf.full():
            self.image_buf.get()
        self.image_buf.put(image_msg)
        self.lock.release()

    def detect(self, image, cfg):
        image = PILImage.fromarray(image)
        image_size = (image.width, image.height)
        heatmap_size = cfg.heatmap_size

        image = image.resize(self.dataset.image_size, PILImage.NEAREST)
        images = self.dataset.transforms(image).to(self.device)[None]
        _, _, line_preds, line_scores = self.model(images)
        line_pred = line_preds[0].detach().cpu().numpy()
        line_score = line_scores[0].detach().cpu().numpy()
        line_pred = line_pred[line_score > cfg.score_thresh]

        sx, sy = 1.0 * image_size[0] / heatmap_size[0], 1.0 * image_size[1] / heatmap_size[1]
        line_pred[:, :, 0] *= sx
        line_pred[:, :, 1] *= sy
        return line_pred.reshape(-1, 4)

    def start(self, cfg):

        while not rospy.is_shutdown():

            image_msg = None
            self.lock.acquire()
            if not self.image_buf.empty():
                image_msg = self.image_buf.get()
            self.lock.release()

            if image_msg:
                image = self.br.imgmsg_to_cv2(image_msg)
                if len(image.shape) == 2:  # gray image
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                if self.cameraMatrix is not None:
                    image = cv2.undistort(image, self.cameraMatrix, self.distCoeffs, newCameraMatrix=self.cameraMatrix)

                lines = self.detect(image, cfg)
                lines2d_msg = lines2d(
                    header=image_msg.header, startx=lines[:, 0], starty=lines[:, 1], endx=lines[:, 2], endy=lines[:, 3])
                self.pub_lines.publish(lines2d_msg)
                if self.publish_image:
                    for line in lines:
                        cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
                        cv2.circle(image, (line[0], line[1]), radius=3, color=(255, 0, 0), thickness=2)
                        cv2.circle(image, (line[2], line[3]), radius=3, color=(255, 0, 0), thickness=2)
                    self.pub_image.publish(self.br.cv2_to_imgmsg(image, "bgr8"))

            self.loop_rate.sleep()


if __name__ == "__main__":
    rospy.init_node('ulsd')

    config_file = rospy.get_param('~config_file')
    sensor_file = rospy.get_param('~sensor_file', '')
    cfg = CfgNode.load_cfg(open(config_file))
    cfg.publish_image = rospy.get_param('~publish_image', False)
    cfg.freeze()
    print(cfg)

    root_path = rospy.get_param('~root_path')
    os.chdir(root_path)

    node = Node(cfg, sensor_file)
    node.start(cfg)

