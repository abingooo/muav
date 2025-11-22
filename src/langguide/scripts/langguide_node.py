#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入必要的Python库
import os

# 导入图像处理库
import cv2
import numpy as np

# 导入ROS相关库
import rospkg
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from nav_msgs.msg import Odometry

try:
    from cv_bridge import CvBridge, CvBridgeError
except ImportError:  # 当cv_bridge不可用时提供备选方案
    CvBridge = None  # type: ignore
    CvBridgeError = RuntimeError  # type: ignore


class LangGuideNode:
    """订阅 RGB-D 话题、缓存最新图像并周期发布状态的基础节点。"""

    def __init__(self):
        # 允许通过参数调整订阅/发布话题
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.status_topic = rospy.get_param("~status_topic", "langguide/status")
        # VINS-Fusion IMU传播位姿话题
        self.odom_topic = rospy.get_param("~odom_topic", "/vins_fusion/imu_propagate")
        # 语言指令话题
        self.lang_cmd_topic = rospy.get_param("~lang_cmd_topic", "/lang_cmd")

        status_rate = rospy.get_param("~status_rate", 10.0)
        if status_rate <= 0.0:
            rospy.logwarn("status_rate <= 0 detected, resetting to 10 Hz")
            status_rate = 10.0

        # 数据可用性标记与缓存
        self.have_color = False
        self.have_depth = False
        self.have_odometry = False
        self.have_cmd = False
        self.warned_missing = False
        self.last_color = None
        self.last_depth = None
        self.last_odometry = None
        self.position = None  # 存储位置信息
        self.orientation = None  # 存储姿态四元数信息

        queue_size = rospy.get_param("~queue_size", 10)
        self.bridge = CvBridge() if CvBridge else None
        self.bridge_active = self.bridge is not None
        if not self.bridge_active:
            rospy.logwarn("cv_bridge unavailable, falling back to manual decoding.")
        pkg_path = rospkg.RosPack().get_path("langguide")
        workspace_root = os.path.abspath(os.path.join(pkg_path, "..", ".."))
        default_debug_dir = os.path.join(workspace_root, "debug")
        self.debug_dir = rospy.get_param("~debug_dir", default_debug_dir)
        self._ensure_debug_dir()
        self.color_path = os.path.join(self.debug_dir, "color_latest.jpg")
        self.depth_npy_path = os.path.join(self.debug_dir, "depth_latest.npy")
        self.depth_png_path = os.path.join(self.debug_dir, "depth_latest.jpg")
        # 添加伪彩色深度图像保存路径
        self.depth_color_path = os.path.join(self.debug_dir, "depth_color_latest.png")
        
        
        # 订阅彩色与深度图像
        self.color_sub = rospy.Subscriber(
            self.color_topic, Image, self._color_cb, queue_size=queue_size
        )
        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self._depth_cb, queue_size=queue_size
        )
        # 订阅VINS-Fusion的IMU传播位姿话题
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=queue_size
        )
        # 订阅语言指令话题
        self.lang_cmd_sub = rospy.Subscriber(
            self.lang_cmd_topic, String, self._lang_cmd_cb, queue_size=queue_size
        )
        # 发布状态字符串供其他模块监控
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=False
        )
        self.status_srv = rospy.Service("~data_status", Trigger, self._handle_status)

        # 定时器相当于 10Hz 主循环
        self.timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / status_rate), self._timer_cb
        )

        rospy.loginfo(
            "LangGuide node subscribed to %s, %s, %s, and %s, publishing %s at %.1f Hz",
            self.color_topic,
            self.depth_topic,
            self.odom_topic,
            self.lang_cmd_topic,
            self.status_topic,
            status_rate,
        )

    def _color_cb(self, msg):
        self.last_color = msg
        self.have_color = True
        if not self._save_color_via_bridge(msg):
            array = self._decode_color_manual(msg)
            if array is not None:
                try:
                    cv2.imwrite(self.color_path, array)
                except cv2.error as err:
                    rospy.logwarn_throttle(5.0, "Failed to save color image: %s", err)
            else:
                rospy.logwarn_throttle(
                    5.0, "Unsupported color encoding: %s", msg.encoding
                )

    def _depth_cb(self, msg):
        self.last_depth = msg
        self.have_depth = True
        depth_array = self._save_depth_via_bridge(msg)
        if depth_array is None:
            depth_array = self._decode_depth_manual(msg)
        if depth_array is None:
            rospy.logwarn_throttle(5.0, "Unsupported depth encoding: %s", msg.encoding)
            return
        try:
            np.save(self.depth_npy_path, depth_array)
            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            cv2.imwrite(self.depth_png_path, depth_colored)
            
        except (ValueError, cv2.error) as err:
            rospy.logwarn_throttle(5.0, "Failed to save depth data: %s", err)
    
    def _timer_cb(self, _event):
        if not (self.have_color and self.have_depth):
            if not self.warned_missing:
                self.warned_missing = True
                self.status_pub.publish(
                    String(data="Waiting for both color and depth images...")
                )
            return

        self.warned_missing = False
        self.status_pub.publish(String(data=self._compose_status()))
    
    def _odom_cb(self, msg):
        """处理VINS-Fusion的IMU传播位姿数据"""
        self.last_odometry = msg
        self.have_odometry = True
        # 提取位置信息
        self.position = msg.pose.pose.position
        # 提取姿态四元数信息
        self.orientation = msg.pose.pose.orientation
        rospy.logdebug("Received odometry data: pos=(%f, %f, %f), orient=(%f, %f, %f, %f)",
                     self.position.x, self.position.y, self.position.z,
                     self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w)

    # 在_lang_cmd_cb方法中，将原来的subprocess调用替换为函数调用
    
    def _lang_cmd_cb(self, msg):
        """处理语言指令"""
        self.last_lang_cmd = msg
        self.have_cmd = True
        rospy.loginfo("Received language command: %s", msg.data)
        
        # 调用egovlm的模块函数处理语言指令
        try:
            import sys
            import os
            # 添加egovlm目录到Python路径
            egovlm_path = os.path.join(os.path.dirname(__file__), 'egovlm')
            if egovlm_path not in sys.path:
                sys.path.append(egovlm_path)
            
            # 将控制指令保存到commend文件中，供egovlm使用
            commend_file_path = os.path.join(egovlm_path, 'commend')
            with open(commend_file_path, 'w') as f:
                f.write(msg.data)
            
            rospy.loginfo("Command saved to %s", commend_file_path)
            
            # 直接导入并调用egovlm_module中的process_command函数
            # 使用绝对导入路径来避免模块导入问题
            import sys
            import os
            
            # 获取当前脚本所在目录
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            # 添加egovlm目录到Python路径
            egovlm_dir = os.path.join(current_script_dir, 'egovlm')
            if egovlm_dir not in sys.path:
                sys.path.append(egovlm_dir)
            
            # 现在尝试导入egovlm_module
            try:
                from egovlm_module import process_command
            except ImportError:
                # 如果直接导入失败，尝试使用__import__函数
                import importlib.util
                spec = importlib.util.spec_from_file_location("egovlm_module", os.path.join(egovlm_dir, "egovlm_module.py"))
                egovlm_module = importlib.util.module_from_spec(spec)
                sys.modules["egovlm_module"] = egovlm_module
                spec.loader.exec_module(egovlm_module)
                process_command = egovlm_module.process_command
            
            # 在后台线程中执行，避免阻塞ROS节点
            from threading import Thread
            def run_egovlm():
                try:
                    # 切换工作目录到egovlm目录
                    original_dir = os.getcwd()
                    os.chdir(egovlm_path)
                    
                    # 将ROS Image消息转换为numpy数组
                    try:
                        # 转换彩色图像
                        if self.bridge_active and self.last_color is not None:
                            color_array = self.bridge.imgmsg_to_cv2(self.last_color, "bgr8")
                        else:
                            color_array = self._decode_color_manual(self.last_color)
                            
                        # 转换深度图像
                        if self.bridge_active and self.last_depth is not None:
                            depth_array = self.bridge.imgmsg_to_cv2(self.last_depth, "32FC1")
                        else:
                            depth_array = self._decode_depth_manual(self.last_depth)
                        
                        # 检查转换结果
                        if color_array is None or depth_array is None:
                            rospy.logerr("Failed to convert image data to numpy arrays")
                            return
                            
                        rospy.loginfo("Image data converted successfully")
                        
                        # 调用处理函数
                        rospy.loginfo("Calling egovlm process_command function")
                        rospy.loginfo("color_array shape: %s", color_array.shape)
                        rospy.loginfo("depth_array shape: %s", depth_array.shape)
                        result = process_command(msg.data, color_array, depth_array)
                    except Exception as e:
                        rospy.logerr("Error converting image data: %s", str(e))
                        return
                    
                    # 处理结果
                    if result['success']:
                        rospy.loginfo("Egovlm processing successful")
                        # 这里可以添加对path_points等结果的进一步处理
                        rospy.loginfo("Path points: %s", result['path_points'])
                    else:
                        rospy.logerr("Egovlm processing failed: %s", result['message'])
                        
                except ImportError as e:
                    rospy.logerr("Failed to import egovlm_module: %s", str(e))
                except Exception as e:
                    rospy.logerr("Error running egovlm: %s", str(e))
                finally:
                    # 恢复原来的工作目录
                    os.chdir(original_dir)
            
            # 启动后台线程执行egovlm
            thread = Thread(target=run_egovlm)
            thread.daemon = True  # 设为守护线程，ROS节点关闭时自动终止
            thread.start()
            
        except Exception as e:
            rospy.logerr("Failed to call egovlm module: %s", str(e))
        
        # self.have_cmd = False

    def _handle_status(self, _req):
        resp = TriggerResponse()
        if not (self.have_color and self.have_depth):
            resp.success = False
            resp.message = "Images not received yet"
            return resp

        resp.success = True
        resp.message = self._compose_status()
        return resp

    def _compose_status(self):
        now = rospy.Time.now()
        color_age = (now - self.last_color.header.stamp).to_sec()
        depth_age = (now - self.last_depth.header.stamp).to_sec()
        
        # 基本状态信息
        msg_parts = [
            f"Color stamp: {self.last_color.header.stamp.to_sec():.6f}",
            f" ({self.last_color.width}x{self.last_color.height}), ",
            f"Depth stamp: {self.last_depth.header.stamp.to_sec():.6f}",
            f" ({self.last_depth.width}x{self.last_depth.height}), "
        ]
        
        # 如果有里程计信息，添加到状态中
        if self.have_odometry and self.last_odometry:
            odom_age = (now - self.last_odometry.header.stamp).to_sec()
            msg_parts.extend([
                f"Odom stamp: {self.last_odometry.header.stamp.to_sec():.6f}, ",
                f"pos: ({self.position.x:.3f}, {self.position.y:.3f}, {self.position.z:.3f}), ",
                f"orient: ({self.orientation.w:.3f}, {self.orientation.x:.3f}, {self.orientation.y:.3f}, {self.orientation.z:.3f}), ",
                f"odom age: {odom_age:.3f}s, "
            ])
        if self.have_cmd and self.last_lang_cmd:
            msg_parts.extend([
                f"Lang cmd: {self.last_lang_cmd.data}"
            ])
            self.have_cmd = False
        # 添加图像数据年龄信息
        msg_parts.extend([
            f"color age: {color_age:.3f}s, depth age: {depth_age:.3f}s"
        ])
        
        return "".join(msg_parts)

    def _ensure_debug_dir(self):
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
            rospy.loginfo("Created debug directory at %s", self.debug_dir)

    @staticmethod
    def _depth_to_uint16(depth_array):
        # 支持 float32/float64 等深度编码，转换为毫米的 uint16，便于重复写入 PNG
        depth = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.dtype == np.uint16:
            return depth
        # clip to 0-65.535m before转毫米
        depth_mm = np.clip(depth * 1000.0, 0, 65535)
        return depth_mm.astype(np.uint16)

    def _save_color_via_bridge(self, msg):
        if not self.bridge_active or self.bridge is None:
            return False
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv2.imwrite(self.color_path, cv_image)
            return True
        except Exception as err:  # broad to catch linkage/runtime errors
            rospy.logwarn_throttle(5.0, "cv_bridge failed for color image: %s", err)
            self.bridge_active = False
            return False

    def _save_depth_via_bridge(self, msg):
        if not self.bridge_active or self.bridge is None:
            return None
        try:
            depth_array = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            return depth_array
        except Exception as err:
            rospy.logwarn_throttle(5.0, "cv_bridge failed for depth image: %s", err)
            self.bridge_active = False
            return None

    def _decode_color_manual(self, msg):
        encoding = (msg.encoding or "").lower()
        mapping = {
            "bgr8": (np.uint8, 3),
            "rgb8": (np.uint8, 3),
            "mono8": (np.uint8, 1),
            "mono16": (np.uint16, 1),
        }
        if encoding not in mapping:
            return None
        dtype, channels = mapping[encoding]
        expected = msg.height * msg.width * channels
        flat = np.frombuffer(msg.data, dtype=dtype, count=expected)
        if flat.size != expected:
            rospy.logwarn_throttle(
                5.0,
                "Color image size mismatch (encoding %s, got %d, expected %d)",
                encoding,
                flat.size,
                expected,
            )
            return None
        if channels == 1:
            image = flat.reshape((msg.height, msg.width))
            return image
        image = flat.reshape((msg.height, msg.width, channels))
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _decode_depth_manual(self, msg):
        encoding = (msg.encoding or "").lower()
        mapping = {
            "32fc1": np.float32,
            "16uc1": np.uint16,
            "16sc1": np.int16,
        }
        if encoding not in mapping:
            return None
        dtype = mapping[encoding]
        expected = msg.height * msg.width
        flat = np.frombuffer(msg.data, dtype=dtype, count=expected)
        if flat.size != expected:
            rospy.logwarn_throttle(
                5.0,
                "Depth image size mismatch (encoding %s, got %d, expected %d)",
                encoding,
                flat.size,
                expected,
            )
            return None
        depth = flat.reshape((msg.height, msg.width))
        # convert integer types to float meters for consistent downstream usage
        if dtype == np.uint16:
            return depth.astype(np.float32) / 1000.0
        if dtype == np.int16:
            return depth.astype(np.float32) / 1000.0
        return depth


def main():
    rospy.init_node("langguide_node")
    LangGuideNode()
    rospy.spin()


if __name__ == "__main__":
    main()