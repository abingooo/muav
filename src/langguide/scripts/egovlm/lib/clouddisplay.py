import argparse
import os
import sys
import open3d as o3d

def display_ply_file(file_path, bg_color=None):
    """
    读取并显示PLY文件
    
    参数:
        file_path: PLY文件的路径
        bg_color: 背景颜色，格式为[R, G, B]，值范围为[0, 1]
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return False
        
        # 检查文件扩展名
        if not file_path.lower().endswith('.ply'):
            print(f"错误: 文件 '{file_path}' 不是PLY格式文件")
            return False
        
        print(f"正在读取文件: {file_path}")
        
        # 读取点云数据
        pcd = o3d.io.read_point_cloud(file_path)
        
        # 检查点云是否为空
        if len(pcd.points) == 0:
            print("错误: 点云数据为空")
            return False
        
        print(f"点云包含 {len(pcd.points)} 个点")
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"点云可视化 - {os.path.basename(file_path)}")
        
        # 添加点云到可视化窗口
        vis.add_geometry(pcd)
        
        # 设置可视化参数
        render_option = vis.get_render_option()
        render_option.point_size = 2.0  # 设置点的大小
        
        # 设置背景颜色
        if bg_color is None:
            # 默认背景色：深灰色
            render_option.background_color = [0.05, 0.05, 0.05]
            print("使用默认背景颜色: 深灰色")
        else:
            # 验证颜色值范围
            for c in bg_color:
                if c < 0 or c > 1:
                    print("警告: 颜色值应在[0, 1]范围内，将使用默认值")
                    render_option.background_color = [0.05, 0.05, 0.05]
                    break
            else:
                # 所有颜色值都在有效范围内
                render_option.background_color = bg_color
                print(f"使用自定义背景颜色: R={bg_color[0]:.2f}, G={bg_color[1]:.2f}, B={bg_color[2]:.2f}")
        
        # 添加坐标系（可选）
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        print("\n可视化窗口已打开")
        print("操作提示:")
        print("  - 鼠标左键拖动: 旋转视角")
        print("  - 鼠标右键拖动: 平移视角")
        print("  - 鼠标滚轮: 缩放")
        print("  - 按 'ESC' 或关闭窗口: 退出")
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
        return True
        
    except Exception as e:
        print(f"显示点云时出错: {e}")
        return False

def parse_color_arg(color_str):
    """
    解析颜色参数字符串为RGB列表
    
    参数:
        color_str: 颜色字符串，格式如 "0.5,0.5,0.5"
    
    返回:
        RGB颜色列表，格式为 [R, G, B]
    """
    try:
        # 分割字符串并转换为浮点数
        color_values = list(map(float, color_str.split(',')))
        # 检查是否有三个值
        if len(color_values) != 3:
            raise ValueError("颜色值必须包含三个分量: R, G, B")
        return color_values
    except Exception as e:
        raise argparse.ArgumentTypeError(f"无效的颜色格式: {e}. 正确格式应为逗号分隔的三个浮点数，如 '0.5,0.5,0.5'")

def main():
    """
    主函数，处理命令行参数并显示点云
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='读取并显示PLY格式的点云文件')
    parser.add_argument('file_path', nargs='?', default='./log/point_cloud.ply',
                        help='要显示的PLY文件路径 (默认为 ./log/point_cloud.ply)')
    parser.add_argument('--bgcolor', type=parse_color_arg,
                        help='背景颜色，格式为逗号分隔的三个浮点数 (0-1)，如 "0,0,0" 表示黑色，"1,1,1" 表示白色')
    parser.add_argument('--black', action='store_true', help='使用黑色背景')
    parser.add_argument('--white', action='store_true', help='使用白色背景')
    parser.add_argument('--gray', action='store_true', help='使用深灰色背景 (默认)')
    parser.add_argument('--blue', action='store_true', help='使用蓝色背景')
    parser.add_argument('--green', action='store_true', help='使用绿色背景')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确定背景颜色
    bg_color = None
    if args.bgcolor:
        bg_color = args.bgcolor
    elif args.black:
        bg_color = [0.0, 0.0, 0.0]
    elif args.white:
        bg_color = [1.0, 1.0, 1.0]
    elif args.blue:
        bg_color = [0.0, 0.0, 0.7]
    elif args.green:
        bg_color = [0.0, 0.7, 0.0]
    # 如果没有指定，保持为None，将使用默认值
    
    # 显示点云
    success = display_ply_file(args.file_path, bg_color)
    
    # 根据执行结果设置退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


# # 黑色背景
# python clouddisplay.py ./log/point_cloud_cube.ply --black

# # 白色背景  
# python clouddisplay.py ./log/point_cloud_cube.ply --white

# # 蓝色背景
# python clouddisplay.py ./log/point_cloud_cube.ply --blue

# # 绿色背景
# python clouddisplay.py ./log/point_cloud_cube.ply --green

# # 深灰色背景（与默认相同）
# python clouddisplay.py ./log/point_cloud_cube.ply --gray