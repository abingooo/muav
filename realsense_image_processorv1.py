# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import os
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 全局变量用于存储最新的图像数据
color_image = None
depth_image = None
b_color_received = False
b_depth_received = False

# 创建CvBridge实例用于ROS图像消息和OpenCV图像之间的转换
bridge = CvBridge()

# 彩色图像回调函数
def color_image_callback(data):
    global color_image, b_color_received
    try:
        # 将ROS图像消息转换为OpenCV图像格式 (BGR8)
        color_image = bridge.imgmsg_to_cv2(data, "bgr8")
        b_color_received = True
        # rospy.loginfo("彩色图像已接收，分辨率: %dx%d", color_image.shape[1], color_image.shape[0])
    except Exception as e:
        rospy.logerr("彩色图像转换错误: %s", str(e))

# 深度图像回调函数
def depth_image_callback(data):
    global depth_image, b_depth_received
    try:
        # 将ROS深度图像消息转换为OpenCV图像格式 (32FC1)
        depth_image = bridge.imgmsg_to_cv2(data, "32FC1")
        b_depth_received = True
        # rospy.loginfo("深度图像已接收，分辨率: %dx%d", depth_image.shape[1], depth_image.shape[0])
    except Exception as e:
        rospy.logerr("深度图像转换错误: %s", str(e))

# 保存图像和深度数据的函数
def save_image_and_depth():
    """优化的图像和深度数据保存函数，使用时间戳创建子文件夹"""
    global color_image, depth_image, b_color_received, b_depth_received
    
    try:
        # 创建基于时间戳的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = "./log"
        save_dir = os.path.join(base_dir, timestamp)
        
        # 确保基础目录存在
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            rospy.loginfo(f"已创建基础目录: {base_dir}")
        
        # 创建时间戳子文件夹
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            rospy.loginfo(f"已创建时间戳保存目录: {save_dir}")
        
        # 创建日志文件
        log_file_path = os.path.join(save_dir, "capture_log.txt")
        log_lines = [f"Capture Timestamp: {timestamp}", ""]
        
        # 保存彩色图像
        if b_color_received:
            color_path = os.path.join(save_dir, "color_image.jpg")
            success = cv2.imwrite(color_path, color_image)
            if success:
                rospy.loginfo(f"彩色图像已成功保存: {color_path}")
                log_lines.append(f"Color Image: {os.path.basename(color_path)} - Saved Successfully")
            else:
                rospy.logerr(f"彩色图像保存失败: {color_path}")
                log_lines.append(f"Color Image: {os.path.basename(color_path)} - Save Failed")
        else:
            rospy.logwarn("未收到彩色图像，跳过保存")
            log_lines.append("Color Image: Not Received")
        
        # 保存深度图像和原始数据
        if b_depth_received:
            # 计算并记录深度图像的统计信息
            depth_min = np.min(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 0
            depth_max = np.max(depth_image)
            depth_mean = np.mean(depth_image[depth_image > 0]) if np.any(depth_image > 0) else 0
            depth_stats = f"深度图像统计: 最小值={depth_min*0.001:.2f}m, 最大值={depth_max*0.001:.2f}m, 平均值={depth_mean:.2f}m"
            rospy.loginfo(depth_stats)
            log_lines.append("\n" + depth_stats.replace("深度图像统计: ", "Depth Stats: "))
            log_lines.append(f"Depth Image Shape: {depth_image.shape}")
            
            # 保存深度图像的可视化版本
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            depth_jpg_path = os.path.join(save_dir, "depth_image.jpg")
            success = cv2.imwrite(depth_jpg_path, depth_colored)
            if success:
                rospy.loginfo(f"深度图像可视化版本已成功保存: {depth_jpg_path}")
                log_lines.append(f"Depth Visualization: {os.path.basename(depth_jpg_path)} - Saved Successfully")
            else:
                rospy.logerr(f"深度图像可视化版本保存失败: {depth_jpg_path}")
                log_lines.append(f"Depth Visualization: {os.path.basename(depth_jpg_path)} - Save Failed")
            
            # 保存深度图像的原始数据为.npy文件
            depth_npy_path = os.path.join(save_dir, "depth_image.npy")
            try:
                np.save(depth_npy_path, depth_image)
                rospy.loginfo(f"深度图像原始数据已成功保存: {depth_npy_path}")
                log_lines.append(f"Raw Depth Data: {os.path.basename(depth_npy_path)} - Saved Successfully")
            except Exception as e:
                error_msg = f"深度图像原始数据保存失败: {str(e)}"
                rospy.logerr(error_msg)
                log_lines.append(f"Raw Depth Data: Save Failed - {str(e)}")
        else:
            rospy.logwarn("未收到深度图像，跳过保存")
            log_lines.append("Depth Image: Not Received")
        
        # 写入日志文件
        try:
            with open(log_file_path, 'w') as log_file:
                log_file.write('\n'.join(log_lines))
            rospy.loginfo(f"捕获日志已保存: {log_file_path}")
        except Exception as e:
            rospy.logerr(f"写入日志文件失败: {str(e)}")
            
        return True
    except Exception as e:
        rospy.logerr(f"保存图像过程中发生错误: {str(e)}")
        return False

# 显示和处理图像的函数
def process_images():
    global color_image, depth_image, b_color_received, b_depth_received
    rospy.loginfo("开始处理图像")
    
    # 执行保存操作
    save_image_and_depth()
    
    # 显示等待用户按键
    rospy.loginfo("等待用户按键继续...")
    cv2.waitKey(0)
    
    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    rospy.loginfo("图像窗口已关闭")

def main():
    # 初始化ROS节点
    rospy.init_node('realsense_image_processor', anonymous=True)
    rospy.loginfo("Realsense图像处理器节点已启动")
    
    # 订阅彩色图像话题
    color_sub = rospy.Subscriber("/camera/color/image_raw", Image, color_image_callback)
    # 检查订阅是否成功
    if color_sub:
        rospy.loginfo("已成功订阅彩色图像话题: /camera/color/image_raw")
    else:
        rospy.logerr("订阅彩色图像话题失败: /camera/color/image_raw")
    
    # 订阅深度图像话题（修复话题名称，添加开头的斜杠）
    depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_image_callback)
    # 检查订阅是否成功
    if depth_sub:
        rospy.loginfo("已成功订阅深度图像话题: /camera/aligned_depth_to_color/image_raw")
    else:
        rospy.logerr("订阅深度图像话题失败: /camera/aligned_depth_to_color/image_raw")
    
    # 检查是否至少有一个话题订阅成功
    if not color_sub and not depth_sub:
        rospy.logfatal("所有图像话题订阅失败，请检查话题名称和Realsense节点是否正常运行！")
        rospy.signal_shutdown("话题订阅失败")
        return
    
    try:
        # 等待图像数据接收，设置超时时间为5秒
        timeout = rospy.Duration(5.0)
        start_time = rospy.Time.now()
        
        # 等待直到收到至少一个图像或超时
        while not (b_color_received or b_depth_received):
            if rospy.Time.now() - start_time > timeout:
                # 根据订阅状态给出更详细的超时信息
                if color_sub and not b_color_received:
                    rospy.logerr("超时警告: 已订阅彩色图像话题但未收到数据")
                if depth_sub and not b_depth_received:
                    rospy.logerr("超时警告: 已订阅深度图像话题但未收到数据")
                
                # 如果没有收到任何图像数据，给出错误提示
                if not b_color_received and not b_depth_received:
                    rospy.logerr("等待图像超时，未收到任何图像数据，请检查Realsense相机是否正常工作以及话题是否正确发布！")
                break
            rospy.loginfo("等待图像数据中...")
            rospy.sleep(0.5)  # 每0.5秒检查一次
        
        # 开始处理图像
        process_images()
    except rospy.ROSInterruptException:
        rospy.logerr("处理过程中发生ROS中断异常")
    except Exception as e:
        rospy.logerr(f"处理过程中发生错误: {str(e)}")
    
    rospy.loginfo("图像处理器节点已关闭")

if __name__ == '__main__':
    main()