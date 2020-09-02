#!/usr/bin/env python
import rospy
import torch
import parsing
import time
import datetime
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from hawp.msg import lines2d
import cv2
import numpy as np
import logging
from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset.build import build_transform
from parsing.detector import WireframeDetector
from parsing.utils.logger import setup_logger
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer



threshold = 0.97

class Nodo(object):
    def __init__(self, cfg):
        # Params
        self.image = None
        self.header = None
        self.br = CvBridge()
        self.pub_img_set = True
        self.image_topic = cfg.IMAGE_TOPIC

        self.logger = logging.getLogger("hawp.testing")
        self.device = cfg.MODEL.DEVICE
        self.model = WireframeDetector(cfg)
        self.model = self.model.to(self.device)
        self.loop_rate = rospy.Rate(200)

        # Publishers
        self.pub = rospy.Publisher('/hawp/Lines2d', lines2d, queue_size=1000)
        if self.pub_img_set:
            self.pub_image = rospy.Publisher('/hawp/feature_image', Image, queue_size=1000)

        # Subscribers
        rospy.Subscriber(self.image_topic, Image, self.callback)

        # camera parameters
        # self.mtx = np.array([[cfg.projection_parameters.fx, 0, cfg.projection_parameters.cx],
        #                      [0, cfg.projection_parameters.fy, cfg.projection_parameters.cy],
        #                      [0, 0, 1]])
        # self.dist = np.array([cfg.distortion_parameters.k1, cfg.distortion_parameters.k2, cfg.distortion_parameters.p1,
        #                       cfg.distortion_parameters.p2])
        # self.width = cfg.width
        # self.height = cfg.height
        # self.newmtx, self.validpixROI = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.width, self.height), 0,
        #                                                               (self.width, self.height))
        # print(self.newmtx)

        #hawp
        checkpointer = DetectronCheckpointer(cfg,
                                         self.model,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=False,
                                         logger=self.logger)
        _ = checkpointer.load()
        self.transform = build_transform(cfg)
        self.model = self.model.eval()

    def callback(self, msg):
        self.header = msg.header
        self.image = self.br.imgmsg_to_cv2(msg)
        # dst_img=cv2.undistort(img, self.mtx, self.dist, newCameraMatrix=self.newmtx)
        # self.image = img

    def start(self):
        pre_msg_time = rospy.Time(0)
        while not rospy.is_shutdown():
            if self.image is not None:
                msg_time = self.header.stamp
                if msg_time > pre_msg_time:
                    pre_msg_time = msg_time
                    if len(self.image.shape) == 2:  # gray image
                        img = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
                        img_tensor = self.transform(img.astype(float))[None].to(self.device)
                    else:
                        img = self.image.copy()
                        img_tensor = self.transform(img.astype(float))[None].to(self.device)
                    meta = {
                        'filename': 'impath',
                        'height': img.shape[0],
                        'width': img.shape[1],
                    }
                    with torch.no_grad():
                        output, _ = self.model(img_tensor, [meta])
                        output = to_device(output, 'cpu')
                    lines = output['lines_pred'].numpy()
                    scores = output['lines_score'].numpy()
                    idx = scores > threshold
                    lines=lines[idx,:]
                    lines2d_msg = lines2d(
                        header=self.header, startx=lines[:, 0], starty=lines[:, 1], endx=lines[:, 2], endy=lines[:, 3])
                    self.pub.publish(lines2d_msg)
                    if self.pub_img_set:
                        feat_imge = img.copy()
                        for i in range(lines.shape[0]):
                            cv2.line(feat_imge, (lines[i, 0], lines[i, 1]),
                                     (lines[i, 2], lines[i, 3]), (0, 0, 255), 2)
                            cv2.circle(feat_imge, (lines[i, 0], lines[i, 1]), radius=3, color=(255, 0, 0), thickness=2)
                            cv2.circle(feat_imge, (lines[i, 2], lines[i, 3]), radius=3, color=(255, 0, 0),thickness=2)
                        self.pub_image.publish(self.br.cv2_to_imgmsg(feat_imge, "bgr8"))
            self.loop_rate.sleep()

    
if __name__ == "__main__":

    rospy.init_node('hawp')

    config_file=rospy.get_param('~config_file')
    cfg.merge_from_file(config_file)
    cfg.freeze()
    
    # print(cfg)
    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger('hawp', output_dir)
    # logger.info(args)
    logger.info("Loaded configuration file {}".format(config_file))

    my_node=Nodo(cfg)
    my_node.start()

