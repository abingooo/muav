#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机数据可视化Web服务器
用于实时显示RGB图像、深度数据、位置姿态信息和自然语言控制指令
"""
import os
import json
import time
import threading
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory

app = Flask(__name__)

# 配置项
DEBUG_DIR = "/home/uav/lab/muav/debug"
RGB_DIR = os.path.join(DEBUG_DIR, "rgb")
DEPTH_DIR = os.path.join(DEBUG_DIR, "depth")
UPDATE_INTERVAL = 0.1  # 100ms更新一次

# 全局数据存储
current_data = {
    "rgb_image": None,
    "depth_image": None,
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    "last_update_time": None,
    "latest_command": "",
    "status": "初始化中..."
}

# 确保目录存在
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(DEPTH_DIR, exist_ok=True)

def get_latest_file(directory):
    """获取目录中最新的文件"""
    try:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception as e:
        print(f"获取最新文件失败: {e}")
        return None

def load_image(image_path):
    """加载图像"""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return image
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None

def colorize_depth_image(depth_image):
    """将深度图像转换为彩色可视化"""
    if depth_image is None or depth_image.size == 0:
        return None
    
    # 归一化深度值到0-255范围
    min_val = np.min(depth_image)
    max_val = np.max(depth_image)
    if max_val - min_val > 0:
        normalized_depth = 255 * (depth_image - min_val) / (max_val - min_val)
    else:
        normalized_depth = np.zeros_like(depth_image, dtype=np.uint8)
    
    # 转换为彩色图像（使用热力图）
    colorized = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
    return colorized

def update_data():
    """定时更新数据"""
    global current_data
    
    while True:
        try:
            # 获取最新的RGB图像
            latest_rgb = get_latest_file(RGB_DIR)
            if latest_rgb:
                rgb_image = load_image(latest_rgb)
                if rgb_image is not None:
                    current_data["rgb_image"] = rgb_image
                current_data["last_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # 获取最新的深度图像
            latest_depth = get_latest_file(DEPTH_DIR)
            if latest_depth:
                depth_image = load_image(latest_depth)
                if depth_image is not None:
                    # 如果深度图像是单通道的，进行彩色可视化
                    if len(depth_image.shape) == 2:
                        current_data["depth_image"] = colorize_depth_image(depth_image)
                    else:
                        current_data["depth_image"] = depth_image
            
            # 这里可以添加从其他数据源读取位置姿态信息的代码
            # 目前使用模拟数据
            current_data["position"]["x"] += 0.01
            current_data["position"]["y"] += 0.02
            current_data["position"]["z"] += 0.005
            
            # 更新状态
            current_data["status"] = "运行中"
            
        except Exception as e:
            print(f"更新数据时出错: {e}")
            current_data["status"] = f"错误: {str(e)}"
        
        time.sleep(UPDATE_INTERVAL)

def generate_frames(image_type):
    """生成图像帧流"""
    while True:
        # 根据图像类型获取对应的图像数据
        if image_type == "rgb":
            image = current_data.get("rgb_image")
        elif image_type == "depth":
            image = current_data.get("depth_image")
        else:
            # 如果没有对应类型的图像，使用默认占位图像
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, f"{image_type} 图像不可用", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if image is None:
            # 创建默认占位图像
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, f"{image_type} 图像加载失败", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        # 使用multipart/x-mixed-replace格式发送
        yield (b'--frame\r\n' 
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def receive_language_command(command):
    """接收并处理自然语言控制指令"""
    global current_data
    current_data["latest_command"] = command
    print(f"接收到指令: {command}")
    return {"status": "success", "received_command": command}

# Flask路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<image_type>')
def video_feed(image_type):
    """图像流端点"""
    return Response(generate_frames(image_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """状态API"""
    return jsonify({
        "position": current_data["position"],
        "orientation": current_data["orientation"],
        "last_update_time": current_data["last_update_time"],
        "latest_command": current_data["latest_command"],
        "status": current_data["status"]
    })

@app.route('/api/send_command', methods=['POST'])
def api_send_command():
    """发送指令API"""
    data = request.get_json()
    if 'command' in data:
        result = receive_language_command(data['command'])
        return jsonify(result)
    return jsonify({"status": "error", "message": "缺少command参数"})

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)

# 主函数
if __name__ == '__main__':
    # 启动数据更新线程
    update_thread = threading.Thread(target=update_data, daemon=True)
    update_thread.start()
    
    # 启动Flask服务器
    print("启动无人机数据可视化Web服务器...")
    print(f"请访问 http://127.0.0.1:5002 查看可视化界面")
    app.run(host='127.0.0.1', port=5002, threaded=True, debug=True)
