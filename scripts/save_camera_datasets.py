#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Copyright (c) 2022 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import pathlib

import cv2
import cv_bridge
import message_filters
import rospkg
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from tf.transformations import quaternion_matrix
from tf.transformations import unit_vector


class CorrectCameraDatasets(object):

    def __init__(self) -> None:
        self.__path = pathlib.Path(rospkg.RosPack().get_path('ovir_3d'))
        self.__path /= rospy.get_param('~dataset_name', 'dataset')
        video = rospy.get_param('~video_name', 'video')
        self.__path /= video
        self.__path.mkdir(parents=True, exist_ok=True)
        (self.__path / 'color').mkdir(parents=True, exist_ok=True)
        (self.__path / 'depth').mkdir(parents=True, exist_ok=True)
        (self.__path / 'poses').mkdir(parents=True, exist_ok=True)
        depth = rospy.wait_for_message('depth_registered/image_rect', Image)
        msg = rospy.wait_for_message('rgb/camera_info', CameraInfo)
        info = {
            'id': video,
            'im_w': msg.width,
            'im_h': msg.height,
            'depth_scale': 1 if depth.encoding == '32FC1' else 1000,
            'cam_intr': [msg.K[:3], msg.K[3:6], msg.K[6:9]],
        }
        info_path = self.__path / 'config.json'
        info_path.write_text(json.dumps(info, indent=4))
        self.__bridge = cv_bridge.CvBridge()
        self.__count = 0
        rgb_sub = message_filters.Subscriber('rgb/image_rect', Image)
        depth_sub = message_filters.Subscriber('depth_registered/image_rect', Image)
        self.__buffer = tf2_ros.Buffer()
        self.__listener = tf2_ros.TransformListener(self.__buffer)
        self.__sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        self.__sync.registerCallback(self.callback)

    def callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        rgb = self.__bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.__bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        camera_pose = PoseStamped()
        camera_pose.header = rgb_msg.header
        camera_pose.pose.orientation.w = 1.0
        try:
            transform = self.__buffer.lookup_transform('map', camera_pose.header.frame_id, camera_pose.header.stamp,
                                                       rospy.Duration(0.1))

            q = unit_vector([
                transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z,
                transform.transform.rotation.w
            ])
            m = quaternion_matrix(q)
            m[:3, -1] = [
                transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z
            ]
            with open(str(self.__path / 'poses' / f'{self.__count:06d}-pose.txt'), 'w') as f:
                f.write(f'{m[0, 0]} {m[0, 1]} {m[0, 2]} {m[0, 3]}\n' + f'{m[1, 0]} {m[1, 1]} {m[1, 2]} {m[1, 3]}\n'
                        + f'{m[2, 0]} {m[2, 1]} {m[2, 2]} {m[2, 3]}\n' + f'{m[3, 0]} {m[3, 1]} {m[3, 2]} {m[3, 3]}\n')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn('Failed to lookup transform')
            return
        cv2.imwrite(str(self.__path / 'color' / f'{self.__count:06d}-color.jpg'), rgb)
        cv2.imwrite(str(self.__path / 'depth' / f'{self.__count:06d}-depth.png'), depth)

        self.__count += 1


if __name__ == '__main__':
    rospy.init_node('record_dataset')
    _ = CorrectCameraDatasets()

    rospy.spin()
