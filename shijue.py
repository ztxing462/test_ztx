"""
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import lagrange  # 用于创建插值多项式

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class VisualAccessibilityAnalyzer:
    """视觉可达性分析器主类"""
    
    def __init__(self, viewer_position, viewer_direction):
        """
        初始化视觉分析器
        
        参数:
        viewer_position -- 视点三维坐标 (mm)
        viewer_direction -- 视线方向向量
        """
        # 视点参数（单位：毫米）
        self.viewer_pos = np.array(viewer_position)  # 视点位置
        # 视线方向归一化处理
        self.view_dir = np.array(viewer_direction) / np.linalg.norm(viewer_direction)

        # 视野参数（单位：度）
        self.optimal_vert_range = (-15, 15)  # 最佳垂直视野范围
        self.max_vert_range = (-20, 40)      # 最大垂直视野范围
        self.optimal_horiz_range = (-15, 15) # 最佳水平视野范围
        self.max_horiz_range = (-35, 35)     # 最大水平视野范围

        # 临界视角参数（1度）- 人眼最小分辨角
        self.critical_angle = 1.0

        # 视域系数插值点定义 (近界率C与评估值r的映射关系)
        self.C_points_optimal = [0.0, 0.33, 0.67, 1.0]  # 最佳视野区域
        self.r_values_optimal = [1.0, 0.95, 0.85, 0.8]
        
        self.C_points_good = [1.0, 1.35, 1.7, 2.0]      # 良好视野区域
        self.r_values_good = [0.8, 0.7, 0.6, 0.4]
        
        self.C_points_general = [2.0, 2.5, 2.8, 3.0]    # 一般视野区域
        self.r_values_general = [0.4, 0.3, 0.2, 0.1]

        # 物姿系数插值点定义 (视线与法线夹角α与评估值g的映射关系)
        self.alpha_points = [0, 30, 60, 75, 90]  # 夹角α（度）
        self.g_values = [0, 0.4, 0.85, 0.98, 1.0]

        # 视角因素插值点定义 (最大视角γ与评估值A的映射关系)
        self.gamma_points = [0, 3, 6, 30, 50, 85, 120]  # 视角γ（度）
        self.A_values = [0, 0.5, 1.0, 1.0, 0.8, 0.3, 0]

        # 零件几何信息存储
        self.part_vertices = None  # 零件顶点坐标
        self.meshes = None         # 三角网格数据

    def print_part_info(self, vertices):
        """
        输出零件详细信息
        
        参数:
        vertices -- 零件顶点坐标数组 (3×3)
        """
        print("\n=== 零件几何信息 ===")
        # 输出顶点坐标
        print(f"顶点1: ({vertices[0][0]:.1f}, {vertices[0][1]:.1f}, {vertices[0][2]:.1f}) mm")
        print(f"顶点2: ({vertices[1][0]:.1f}, {vertices[1][1]:.1f}, {vertices[1][2]:.1f}) mm")
        print(f"顶点3: ({vertices[2][0]:.1f}, {vertices[2][1]:.1f}, {vertices[2][2]:.1f}) mm")

        # 计算并输出边长
        v0, v1, v2 = vertices
        side1 = np.linalg.norm(v1 - v0)
        side2 = np.linalg.norm(v2 - v1)
        side3 = np.linalg.norm(v0 - v2)
        print(f"边长: {side1:.1f} mm, {side2:.1f} mm, {side3:.1f} mm")

        # 计算并输出面积
        v1_vec = v1 - v0
        v2_vec = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(v1_vec, v2_vec))
        print(f"面积: {area:.1f} mm²")

        # 计算并输出中心点
        center = np.mean(vertices, axis=0)
        print(f"中心点: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm")

        # 计算并输出法向量
        normal = np.cross(v1_vec, v2_vec)
        normal = normal / np.linalg.norm(normal)
        print(f"法向量: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")

        # 判断并输出零件朝向
        if abs(normal[2]) > 0.9:
            orientation = "水平" if normal[2] > 0 else "倒置"
        elif abs(normal[0]) > 0.7:
            orientation = "东/西朝向"
        elif abs(normal[1]) > 0.7:
            orientation = "南/北朝向"
        else:
            orientation = "倾斜"
        print(f"朝向: {orientation}")

    def generate_mesh(self, vertices):
        """
        为大型三角形零件生成三角网格
        
        参数:
        vertices -- 零件顶点坐标数组
        
        返回:
        meshes -- 生成的网格列表
        """
        # 保存零件顶点信息
        self.part_vertices = vertices

        # 计算三角形面积
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))

        # 计算网格尺寸（基于临界视角和视距）
        avg_view_dist = 570  # 平均视距(mm) - 人因工程学推荐值
        # 计算临界视角对应的最小分辨尺寸
        D = 2 * avg_view_dist * math.tan(math.radians(self.critical_angle / 2))
        grid_size = 6 * D  # 取建议范围4D-8D的中间值

        # 计算细分级别（基于面积和网格尺寸）
        num_divisions = max(3, int(math.sqrt(area) / grid_size))
        print(f"网格细分级别: {num_divisions} × {num_divisions} (共{num_divisions ** 2}个网格)")

        # 创建均匀三角网格点
        points = []
        for i in range(num_divisions + 1):
            for j in range(num_divisions + 1 - i):
                # 计算重心坐标
                u = i / num_divisions
                v = j / num_divisions
                w = 1 - u - v
                # 计算网格点坐标（重心坐标插值）
                point = (u * vertices[0] + v * vertices[1] + w * vertices[2])
                points.append(point)

        # 构建三角网格
        meshes = []
        for i in range(num_divisions):
            for j in range(num_divisions - i):
                # 计算网格索引（三角剖分）
                idx1 = i * (num_divisions + 1) - i * (i - 1) // 2 + j
                idx2 = idx1 + 1
                idx3 = idx1 + (num_divisions - i) + 1

                # 创建第一个三角形
                tri_vertices = np.array([points[idx1], points[idx2], points[idx3]])
                
                # 计算三角形属性
                center = np.mean(tri_vertices, axis=0)  # 中心点
                v1 = tri_vertices[1] - tri_vertices[0]  # 边向量1
                v2 = tri_vertices[2] - tri_vertices[0]  # 边向量2
                normal = np.cross(v1, v2)  # 法向量
                area = 0.5 * np.linalg.norm(normal)  # 三角形面积
                normal = normal / np.linalg.norm(normal)  # 法向量归一化

                meshes.append({
                    'vertices': tri_vertices,
                    'center': center,
                    'normal': normal,
                    'area': area
                })

                # 创建第二个三角形（四边形剖分）
                if j < num_divisions - i - 1:
                    idx4 = idx3 + 1
                    tri_vertices2 = np.array([points[idx2], points[idx4], points[idx3]])

                    # 计算第二个三角形属性
                    center2 = np.mean(tri_vertices2, axis=0)
                    v1_2 = tri_vertices2[1] - tri_vertices2[0]
                    v2_2 = tri_vertices2[2] - tri_vertices2[0]
                    normal2 = np.cross(v1_2, v2_2)
                    area2 = 0.5 * np.linalg.norm(normal2)
                    normal2 = normal2 / np.linalg.norm(normal2)

                    meshes.append({
                        'vertices': tri_vertices2,
                        'center': center2,
                        'normal': normal2,
                        'area': area2
                    })

        self.meshes = meshes
        return meshes

    def calculate_boundary_rate(self, mesh):
        """
        计算近界率C和视域区域分类
        
        参数:
        mesh -- 单个网格数据
        
        返回:
        C -- 近界率
        region -- 视域区域分类 ('optimal', 'good', 'general')
        """
        # 计算视线向量（从视点到网格中心）
        view_vector = mesh['center'] - self.viewer_pos
        view_dir = view_vector / np.linalg.norm(view_vector)  # 归一化

        # 计算与中心视线的夹角（度）
        dot_product = np.dot(self.view_dir, view_dir)
        clipped_dot = np.clip(dot_product, -1, 1)  # 确保在[-1,1]范围内
        angle_to_center = np.degrees(np.arccos(clipped_dot))

        # 确定视域区域并计算近界率C
        if angle_to_center <= 15:  # 最佳视野区域
            C = angle_to_center / 15
            region = 'optimal'
        elif angle_to_center <= 35:  # 良好视野区域
            C = 1 + (angle_to_center - 15) / 20
            region = 'good'
        else:  # 一般视野区域
            C = 2 + (angle_to_center - 35) / 25
            region = 'general'

        return C, region

    def calculate_view_field_coeff(self, C, region):
        """
        计算视域系数r
        
        参数:
        C -- 近界率
        region -- 视域区域分类
        
        返回:
        r -- 视域系数 (0-1)
        """
        # 使用拉格朗日插值法计算评估值
        if region == 'optimal':
            poly = lagrange(self.C_points_optimal, self.r_values_optimal)
            return poly(C)
        elif region == 'good':
            poly = lagrange(self.C_points_good, self.r_values_good)
            return poly(C)
        else:
            poly = lagrange(self.C_points_general, self.r_values_general)
            return poly(C)

    def calculate_posture_coeff(self, mesh):
        """
        计算物姿系数g
        
        参数:
        mesh -- 单个网格数据
        
        返回:
        g -- 物姿系数 (0-1)
        """
        # 计算视线向量（从视点到网格中心）
        view_vector = mesh['center'] - self.viewer_pos
        view_dir = view_vector / np.linalg.norm(view_vector)  # 归一化

        # 计算视线与网格法线的夹角α（度）
        cos_alpha = np.dot(mesh['normal'], view_dir)
        # 取绝对值确保角度在0-90度范围内
        alpha = np.degrees(np.arcsin(abs(cos_alpha)))

        # 使用拉格朗日插值法计算评估值
        poly = lagrange(self.alpha_points, self.g_values)
        return poly(alpha)

    def calculate_view_angle_factor(self, part_vertices):
        """
        计算视角因素A（使用最大视角法）
        
        参数:
        part_vertices -- 零件顶点坐标
        
        返回:
        A -- 视角因素 (0-1)
        """
        # 计算视点到零件各顶点的方向向量
        view_vectors = [vertex - self.viewer_pos for vertex in part_vertices]
        view_dirs = [v / np.linalg.norm(v) for v in view_vectors]  # 归一化
        
        # 计算所有点对之间的最大视角
        max_angle = 0
        n = len(view_dirs)
        for i in range(n):
            for j in range(i+1, n):
                # 计算两点方向向量之间的夹角
                dot_product = np.dot(view_dirs[i], view_dirs[j])
                clipped_dot = np.clip(dot_product, -1, 1)  # 确保在[-1,1]范围内
                angle = np.degrees(np.arccos(clipped_dot))
                
                # 更新最大夹角
                if angle > max_angle:
                    max_angle = angle
        
        # 计算视角γ（度）并限制在0-120度范围内
        gamma = max(0, min(max_angle, 120))
        
        # 使用线性插值计算视角因素A
        A = np.interp(gamma, self.gamma_points, self.A_values)
        
        # 确保A在0-1范围内
        return max(0, min(A, 1))

    def evaluate_visual_accessibility(self, part_vertices):
        """
        执行视觉可达性评估（单视点）
        
        参数:
        part_vertices -- 零件顶点坐标
        
        返回:
        评估结果字典
        """
        # 生成零件网格
        meshes = self.generate_mesh(part_vertices)
        total_area = sum(mesh['area'] for mesh in meshes)  # 零件总面积

        # 计算视角因素A（对整个零件）
        A = self.calculate_view_angle_factor(part_vertices)

        # 初始化评估值累加器
        R_sum, G_sum = 0, 0

        # 遍历所有网格进行局部评估
        for mesh in meshes:
            # 计算视域系数r
            C, region = self.calculate_boundary_rate(mesh)
            r = self.calculate_view_field_coeff(C, region)
            R_sum += r * mesh['area']  # 面积加权

            # 计算物姿系数g
            g = self.calculate_posture_coeff(mesh)
            G_sum += g * mesh['area']  # 面积加权

        # 计算视域因素评估值R（面积加权平均）
        R = R_sum / total_area if total_area > 0 else 0

        # 计算物姿因素评估值G（面积加权平均）
        G = G_sum / total_area if total_area > 0 else 0

        # 计算视觉工效量F（公式7）
        F = A * (R + G) / 2

        # 计算视觉可达性总体评估值VI（公式8简化）
        VI = (R + G + A + F) / 4

        return {
            'R': R,       # 视域因素评估值
            'G': G,       # 物姿因素评估值
            'A': A,       # 视角因素评估值
            'F': F,       # 视觉工效量
            'VI': VI,     # 视觉可达性指数
            'total_meshes': len(meshes),      # 总网格数
            'visible_meshes': len(meshes),    # 可见网格数（本实现中全部可见）
            'visible_ratio': 1.0              # 可见面积比例
        }

    def multi_viewpoint_evaluation(self, part_vertices):
        """
        多视点评估（3×3垂直网格）
        
        参数:
        part_vertices -- 零件顶点坐标
        
        返回:
        多视点评估结果列表
        """
        # 创建3x3视点网格（垂直平面）
        grid_points = []
        center = np.mean(part_vertices, axis=0)  # 零件中心点

        grid_size = 400  # 网格大小（mm）

        # 生成垂直网格点（X坐标不变，改变Y和Z坐标）
        for y_offset in [-grid_size, 0, grid_size]:
            for z_offset in [-grid_size, 0, grid_size]:
                pos = self.viewer_pos.copy()
                pos[1] += y_offset  # Y坐标偏移
                pos[2] += z_offset  # Z坐标偏移
                grid_points.append(pos)

        VI_values = []  # 存储各视点VI值
        results = []    # 存储评估结果

        # 遍历所有视点进行评估
        for i, pos in enumerate(grid_points):
            self.viewer_pos = np.array(pos)  # 设置当前视点
            # 执行单视点评估
            eval_result = self.evaluate_visual_accessibility(part_vertices)
            VI_values.append(eval_result['VI'])
            # 保存结果
            results.append({
                'viewpoint': pos,               # 视点位置
                'R': eval_result['R'],           # 视域因素
                'G': eval_result['G'],           # 物姿因素
                'A': eval_result['A'],           # 视角因素
                'F': eval_result['F'],           # 视觉工效量
                'VI': eval_result['VI'],         # 视觉可达性指数
                'visible_ratio': eval_result['visible_ratio']  # 可见面积比例
            })

        # 归一化处理VI值
        min_vi = min(VI_values)
        max_vi = max(VI_values)
        range_vi = max_vi - min_vi if max_vi > min_vi else 1  # 防止除零

        # 量化分级处理
        for i, res in enumerate(results):
            normalized_vi = (res['VI'] - min_vi) / range_vi
            level, label = self.quantize_visual_accessibility(normalized_vi)
            res['normalized_VI'] = normalized_vi  # 归一化VI值
            res['level'] = level                  # 量化等级
            res['label'] = label                  # 等级描述

        return results

    def quantize_visual_accessibility(self, VI):
        """
        视觉可达性量化分级（基于表1标准）
        
        参数:
        VI -- 视觉可达性指数（归一化后）
        
        返回:
        level -- 量化等级 (0-7)
        label -- 等级描述
        """
        # 根据VI值范围确定等级
        if VI < 0.12:
            return 0, "几乎不可达"
        elif VI < 0.25:
            return 1, "严重受限"
        elif VI < 0.37:
            return 2, "中度受限"
        elif VI < 0.50:
            return 3, "受限"
        elif VI < 0.62:
            return 4, "临界可达"
        elif VI < 0.75:
            return 5, "基本可达"
        elif VI < 0.88:
            return 6, "良好可达"
        else:
            return 7, "完全可达"

    def visualize_scene(self, part_vertices, viewpoint=None, save_path=None):
        """
        可视化3D场景
        
        参数:
        part_vertices -- 零件顶点坐标
        viewpoint -- 指定视点位置 (默认使用当前视点)
        save_path -- 图片保存路径 (None则显示)
        """
        if viewpoint is None:
            viewpoint = self.viewer_pos  # 使用当前视点

        # 创建图形窗口
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')  # 3D坐标系

        # 绘制三角形零件
        triangle = np.array([part_vertices[0], part_vertices[1], part_vertices[2]])
        ax.add_collection3d(Poly3DCollection([triangle], alpha=0.5, color='lightblue'))

        # 绘制网格中心点（可视化网格分布）
        if self.meshes:
            centers = np.array([mesh['center'] for mesh in self.meshes])
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='b', s=10, alpha=0.3, label='网格中心')

        # 绘制视点
        ax.scatter(viewpoint[0], viewpoint[1], viewpoint[2], c='r', s=100, marker='o', label='视点')

        # 绘制视线方向
        direction_length = 500  # 方向箭头长度
        ax.quiver(viewpoint[0], viewpoint[1], viewpoint[2],
                  self.view_dir[0] * direction_length,
                  self.view_dir[1] * direction_length,
                  self.view_dir[2] * direction_length,
                  color='g', label='视线方向')

        # 设置坐标轴标签
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('视觉可达性分析3D场景')  # 标题

        # 添加图例
        ax.legend()

        # 设置等比例显示
        max_range = np.array([part_vertices[:, 0].max() - part_vertices[:, 0].min(),
                              part_vertices[:, 1].max() - part_vertices[:, 1].min(),
                              part_vertices[:, 2].max() - part_vertices[:, 2].min()]).max() * 0.5
        mid_x = (part_vertices[:, 0].max() + part_vertices[:, 0].min()) * 0.5
        mid_y = (part_vertices[:, 1].max() + part_vertices[:, 1].min()) * 0.5
        mid_z = (part_vertices[:, 2].max() + part_vertices[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"场景已保存至: {save_path}")
        else:
            plt.show()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 定义大型三角形零件（垂直平面，单位：mm）
    large_triangle = np.array([
        [0, 0, 0],        # 顶点1
        [0, 1000, 0],     # 顶点2（垂直方向）
        [0, 500, 866]     # 顶点3（等边三角形，边长1000mm）
    ])

    # 初始视点位置（零件前方）
    viewer_pos = [-1500, 500, 288.7]  # 视点位置
    viewer_dir = [1, 0, 0]            # 指向零件中心

    # 创建分析器实例
    analyzer = VisualAccessibilityAnalyzer(viewer_pos, viewer_dir)

    # 输出零件几何信息
    analyzer.print_part_info(large_triangle)

    # ===== 单视点评估 =====
    print("\n=== 单视点评估 ===")
    print(f"视点位置: ({viewer_pos[0]}, {viewer_pos[1]}, {viewer_pos[2]}) mm")
    print(f"视线方向: ({viewer_dir[0]:.3f}, {viewer_dir[1]:.3f}, {viewer_dir[2]:.3f})")

    # 执行评估
    eval_result = analyzer.evaluate_visual_accessibility(large_triangle)

    # 输出评估结果
    print(f"\n零件网格总数: {eval_result['total_meshes']}")
    print(f"可见网格数量: {eval_result['visible_meshes']}")
    print(f"可见面积比例: {eval_result['visible_ratio']:.2%}")
    print(f"视域因素评估值 R: {eval_result['R']:.4f}")
    print(f"物姿因素评估值 G: {eval_result['G']:.4f}")
    print(f"视角因素评估值 A: {eval_result['A']:.4f}")
    print(f"视觉工效量 F: {eval_result['F']:.4f}")
    print(f"视觉可达性总体评估 VI: {eval_result['VI']:.4f}")

    # 量化分级
    level, label = analyzer.quantize_visual_accessibility(eval_result['VI'])
    print(f"\n量化分级结果: 等级{level} ({label})")

    # 可视化单视点场景
    analyzer.visualize_scene(large_triangle, save_path="single_view_scene.png")

    # ===== 多视点评估 =====
    print("\n=== 多视点评估与量化分级 ===")

    # 计算网格偏移量（相对于初始视点）
    grid_size = 400  # 网格间距(mm)
    center_x = viewer_pos[0]  # X坐标固定
    center_y = viewer_pos[1]  # Y中心位置
    center_z = viewer_pos[2]  # Z中心位置

    # 创建3×3垂直视点网格
    grid_points = []
    for y_offset in [-grid_size, 0, grid_size]:
        for z_offset in [-grid_size, 0, grid_size]:
            grid_points.append([
                center_x,  # X坐标固定
                center_y + y_offset,  # Y偏移
                center_z + z_offset   # Z偏移
            ])

    # 确保网格中心正好是初始视点
    grid_points[4] = viewer_pos.copy()  # 中心点

    VI_values = []  # 存储VI值
    results = []    # 存储结果

    # 遍历所有视点
    for i, pos in enumerate(grid_points):
        # 设置当前视点位置
        analyzer.viewer_pos = np.array(pos)

        # 设置视线方向指向零件中心
        part_center = np.mean(large_triangle, axis=0)
        view_vector = part_center - np.array(pos)
        analyzer.view_dir = view_vector / np.linalg.norm(view_vector)

        # 执行评估
        eval_result = analyzer.evaluate_visual_accessibility(large_triangle)
        VI_values.append(eval_result['VI'])
        results.append({
            'viewpoint': pos,
            'R': eval_result['R'],
            'G': eval_result['G'],
            'A': eval_result['A'],
            'F': eval_result['F'],
            'VI': eval_result['VI'],
            'visible_ratio': eval_result['visible_ratio'],
            'direction': analyzer.view_dir.copy()  # 保存视线方向
        })

    # 归一化处理
    min_vi = min(VI_values)
    max_vi = max(VI_values)
    range_vi = max_vi - min_vi if max_vi > min_vi else 1

    # 量化分级处理
    for i, res in enumerate(results):
        normalized_vi = (res['VI'] - min_vi) / range_vi
        level, label = analyzer.quantize_visual_accessibility(normalized_vi)
        res['normalized_VI'] = normalized_vi
        res['level'] = level
        res['label'] = label

    # 输出多视点评估结果
    print("\n视点评估结果:")
    for i, res in enumerate(results):
        print(f"视点 {i + 1} [X:{res['viewpoint'][0]:.0f} Y:{res['viewpoint'][1]:.0f} Z:{res['viewpoint'][2]:.0f}]:")
        print(f"  R={res['R']:.4f} G={res['G']:.4f} A={res['A']:.4f} F={res['F']:.4f}")
        print(f"  VI={res['VI']:.4f} -> 归一化VI={res['normalized_VI']:.4f}")
        print(f"  量化等级: {res['level']} ({res['label']})")
        print(f"  可见面积: {res['visible_ratio']:.2%}\n")

    # 输出最终分级结果
    print("=== 最终量化分级 ===")
    levels = [res['level'] for res in results]
    print(f"各视点等级: {levels}")
    print(f"最高等级: {max(levels)} ({analyzer.quantize_visual_accessibility(max(levels)/10)[1]})")
    print(f"平均等级: {np.mean(levels):.2f}")

    # ===== 可视化多视点场景 =====
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制零件
    triangle = np.array([large_triangle[0], large_triangle[1], large_triangle[2]])
    ax.add_collection3d(Poly3DCollection([triangle], alpha=0.5, color='lightblue'))

    # 绘制视点（不同颜色）
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for i, res in enumerate(results):
        pos = res['viewpoint']
        ax.scatter(pos[0], pos[1], pos[2], c=[colors[i]], s=100, label=f'视点 {i + 1} (等级{res["level"]})')

        # 绘制视线方向
        direction_length = 300
        ax.quiver(pos[0], pos[1], pos[2],
                  res['direction'][0] * direction_length,
                  res['direction'][1] * direction_length,
                  res['direction'][2] * direction_length,
                  color=colors[i], alpha=0.7)

    # 设置坐标和标题
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('多视点评估场景（垂直视点网格）')
    ax.legend()  # 图例

    # 设置等比例显示范围
    max_range = 2000
    mid_x = viewer_pos[0]
    mid_y = viewer_pos[1]
    mid_z = viewer_pos[2]
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    # 保存场景图
    plt.savefig("multi_view_scene.png", dpi=300)
    print("\n多视点场景已保存至: multi_view_scene.png")

    # ===== 绘制量化等级分布图 =====
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = [res['level'] for res in results]
    ax.bar(range(1, 10), levels, color=colors)  # 柱状图

    # 添加等级标签
    for i, level in enumerate(levels):
        ax.text(i + 1, level + 0.1, str(level), ha='center')

    # 设置图表属性
    ax.set_xlabel('视点编号')
    ax.set_ylabel('量化等级')
    ax.set_title('各视点视觉可达性量化等级（垂直视点网格）')
    ax.set_ylim(0, 8)
    ax.set_xticks(range(1, 10))
    ax.grid(True, linestyle='--', alpha=0.7)  # 网格线

    # 保存分布图
    plt.savefig("quantization_levels.png", dpi=300)
    print("量化等级分布图已保存至: quantization_levels.png")