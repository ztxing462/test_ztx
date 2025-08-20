import numpy as np
import math
from scipy.spatial.transform import Rotation  # 用于处理3D旋转
import random
import pandas as pd  # 用于结果分析和Excel输出
import trimesh  # 用于3D网格处理和碰撞检测
from scipy.spatial import ConvexHull  # 凸包计算（当前未使用）
from collections import defaultdict  # 高效字典操作（当前未使用）

# ====================== 几何模型基类 ======================
class GeometricModel:
    """几何模型的抽象基类，定义所有几何对象的通用接口"""
    def __init__(self, position, color=(0.5, 0.5, 0.5), is_target=False):
        """
        初始化几何模型
        :param position: 模型中心位置 [x, y, z]
        :param color: 可视化颜色 (R, G, B)
        :param is_target: 是否为维修目标零件
        """
        self.position = np.array(position, dtype=float)
        self.color = color
        self.is_target = is_target  # 标记是否为需要维修的目标

    def get_bounding_box(self):
        """获取轴对齐包围盒(AABB)"""
        raise NotImplementedError("此方法需在子类中实现")
    
    def get_aabb(self):
        """获取AABB的最小和最大角点"""
        min_corner, max_corner = self.get_bounding_box()
        return min_corner, max_corner
    
    def get_obb(self):
        """获取有向包围盒(OBB)"""
        return trimesh.bounds.oriented_bounds(self.get_mesh())

    def contains_point(self, point):
        """判断点是否在模型内部"""
        raise NotImplementedError("此方法需在子类中实现")

    def get_mesh(self):
        """获取模型的三角网格表示"""
        raise NotImplementedError("此方法需在子类中实现")


# ====================== 长方体模型 ======================
class BoxModel(GeometricModel):
    """长方体几何模型，用于表示设备外壳等"""
    def __init__(self, position, size, color=(0.5, 0.5, 0.5), is_target=False):
        """
        初始化长方体
        :param size: 长宽高尺寸 [length, width, height]
        """
        super().__init__(position, color, is_target)
        self.size = np.array(size, dtype=float)

    def get_bounding_box(self):
        """计算轴对齐包围盒(AABB)"""
        half_size = self.size / 2
        min_corner = self.position - half_size
        max_corner = self.position + half_size
        return min_corner, max_corner

    def contains_point(self, point):
        """检测点是否在长方体内部"""
        min_corner, max_corner = self.get_bounding_box()
        return np.all(point >= min_corner) and np.all(point <= max_corner)

    def get_mesh(self):
        """生成长方体的网格模型"""
        return trimesh.creation.box(self.size)


# ====================== 圆柱体模型 ======================
class CylinderModel(GeometricModel):
    """圆柱体几何模型，用于表示维修目标零件"""
    def __init__(self, position, height, radius, color=(1, 0, 0), is_target=True):
        """
        初始化圆柱体
        :param height: 圆柱高度
        :param radius: 圆柱半径
        """
        super().__init__(position, color, is_target)
        self.height = height
        self.radius = radius
        self.axis = np.array([0, 0, 1])  # 默认垂直方向

    def get_bounding_box(self):
        """计算轴对齐包围盒(AABB)"""
        min_corner = self.position - np.array([self.radius, self.radius, 0])
        max_corner = self.position + np.array([self.radius, self.radius, self.height])
        return min_corner, max_corner

    def contains_point(self, point):
        """检测点是否在圆柱体内部"""
        local_point = point - self.position
        # Z轴方向检查
        if local_point[2] < 0 or local_point[2] > self.height:
            return False
        # XY平面径向距离检查
        radial_dist = np.linalg.norm(local_point[:2])
        return radial_dist <= self.radius

    def get_mesh(self):
        """生成圆柱体的网格模型"""
        return trimesh.creation.cylinder(radius=self.radius, height=self.height)


# ====================== 工具模型 (扳手) ======================
class WrenchToolModel:
    """扳手工具模型，包含手柄和操作头"""
    def __init__(self, handle_length=0.3, head_diameter=0.06):
        """
        初始化扳手模型
        :param handle_length: 手柄长度 (米)
        :param head_diameter: 操作头直径 (米)
        """
        self.handle_length = handle_length
        self.head_diameter = head_diameter
        # 定义工具的操作区域（用于碰撞检测）
        self.operational_region = [
            np.array([-0.02, -0.02, -0.02]),
            np.array([self.handle_length + 0.02, 0.02, 0.02])
        ]
        self.color = (0.8, 0.2, 0.2)  # 工具颜色
        self.mesh = self._create_mesh()  # 创建网格模型

    def _create_mesh(self):
        """构建扳手的网格模型"""
        # 创建手柄部分（圆柱体）
        handle = trimesh.creation.cylinder(radius=0.01, height=self.handle_length)
        handle.apply_translation([self.handle_length / 2, 0, 0])
        # 创建操作头部分（短圆柱）
        head = trimesh.creation.cylinder(radius=self.head_diameter / 2, height=0.02)
        head.apply_translation([-0.015, 0, 0])
        # 合并两部分网格
        return handle + head

    def transform(self, position, rotation):
        """
        将工具模型变换到指定位置和方向
        :param position: 世界坐标系中的位置 [x, y, z]
        :param rotation: 欧拉角旋转 [pitch, yaw, roll] (度)
        :return: 变换后的网格
        """
        # 计算平移向量
        translation = position - np.array([0, 0, 0])
        # 创建旋转矩阵
        rot_matrix = trimesh.transformations.euler_matrix(
            np.deg2rad(rotation[0]),
            np.deg2rad(rotation[1]),
            np.deg2rad(rotation[2]),
        )
        # 应用变换
        transformed_mesh = self.mesh.copy()
        transformed_mesh.apply_transform(rot_matrix)
        transformed_mesh.apply_translation(translation)
        return transformed_mesh
    
    def get_obb(self, position, rotation):
        """获取有向包围盒(OBB)"""
        transformed_mesh = self.transform(position, rotation)
        return trimesh.bounds.oriented_bounds(transformed_mesh)
    
    def get_aabb(self, position, rotation):
        """获取轴对齐包围盒(AABB)"""
        transformed_mesh = self.transform(position, rotation)
        bounds = transformed_mesh.bounds
        return bounds[0], bounds[1]


# ====================== 人体模型 ======================
class HumanModel(GeometricModel):
    """人体模型，简化表示为长方体"""
    def __init__(self, position, size, color=(0.9, 0.7, 0.5)):
        """
        初始化人体模型
        :param size: 人体尺寸 [width, depth, height]
        """
        super().__init__(position, color)
        self.size = np.array(size, dtype=float)
        # 肩部高度（用于工具定位参考）
        self.shoulder_height = position[2] + size[2] * 0.8

    def get_bounding_box(self):
        """计算轴对齐包围盒(AABB)"""
        half_size = self.size / 2
        min_corner = self.position - np.array([half_size[0], half_size[1], 0])
        max_corner = self.position + np.array([half_size[0], half_size[1], self.size[2]])
        return min_corner, max_corner

    def contains_point(self, point):
        """检测点是否在人体模型内部"""
        min_corner, max_corner = self.get_bounding_box()
        return np.all(point >= min_corner) and np.all(point <= max_corner)

    def get_mesh(self):
        """生成长方体网格表示人体"""
        return trimesh.creation.box(self.size)


# ====================== 八叉树空间分解 ======================
class OctreeNode:
    """八叉树节点，用于空间分区加速碰撞检测"""
    def __init__(self, bounds, depth=0, max_depth=4, max_objects=8):
        """
        初始化八叉树节点
        :param bounds: 节点边界 [min_x, min_y, min_z, max_x, max_y, max_z]
        :param depth: 当前节点深度
        :param max_depth: 最大递归深度
        :param max_objects: 节点最大容纳对象数
        """
        self.bounds = bounds
        self.depth = depth
        self.max_depth = max_depth
        self.max_objects = max_objects
        self.children = []  # 子节点列表
        self.objects = []   # 当前节点包含的对象
        self.is_divided = False  # 是否已分割
        # 计算节点中心
        self.center = np.array([
            (bounds[0] + bounds[3]) / 2,
            (bounds[1] + bounds[4]) / 2,
            (bounds[2] + bounds[5]) / 2
        ])
        # 计算节点尺寸
        self.size = np.array([
            bounds[3] - bounds[0],
            bounds[4] - bounds[1],
            bounds[5] - bounds[2]
        ])
    
    def subdivide(self):
        """将当前节点分割为8个子节点"""
        if self.depth >= self.max_depth or self.is_divided:
            return
            
        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds
        mid_x, mid_y, mid_z = self.center
        
        # 创建8个子节点
        self.children = [
            # 前下左
            OctreeNode([min_x, min_y, min_z, mid_x, mid_y, mid_z], self.depth + 1, self.max_depth, self.max_objects),
            # 前下右
            OctreeNode([min_x, min_y, mid_z, mid_x, mid_y, max_z], self.depth + 1, self.max_depth, self.max_objects),
            # 前上左
            OctreeNode([min_x, mid_y, min_z, mid_x, max_y, mid_z], self.depth + 1, self.max_depth, self.max_objects),
            # 前上右
            OctreeNode([min_x, mid_y, mid_z, mid_x, max_y, max_z], self.depth + 1, self.max_depth, self.max_objects),
            # 后下左
            OctreeNode([mid_x, min_y, min_z, max_x, mid_y, mid_z], self.depth + 1, self.max_depth, self.max_objects),
            # 后下右
            OctreeNode([mid_x, min_y, mid_z, max_x, mid_y, max_z], self.depth + 1, self.max_depth, self.max_objects),
            # 后上左
            OctreeNode([mid_x, mid_y, min_z, max_x, max_y, mid_z], self.depth + 1, self.max_depth, self.max_objects),
            # 后上右
            OctreeNode([mid_x, mid_y, mid_z, max_x, max_y, max_z], self.depth + 1, self.max_depth, self.max_objects)
        ]
        
        self.is_divided = True
        
        # 将当前节点对象分配到子节点
        for obj in self.objects:
            self._insert_into_child(obj)
        self.objects = []  # 清空当前节点对象
    
    def _insert_into_child(self, obj):
        """将对象插入到合适的子节点"""
        for child in self.children:
            if child._aabb_intersects_object(obj):
                child.insert(obj)
    
    def _aabb_intersects_object(self, obj):
        """检测对象AABB是否与当前节点相交"""
        if hasattr(obj, 'get_aabb'):
            obj_min, obj_max = obj.get_aabb()
        else:
            if isinstance(obj, BoxModel) or isinstance(obj, CylinderModel) or isinstance(obj, HumanModel):
                min_corner, max_corner = obj.get_bounding_box()
                obj_min = min_corner
                obj_max = max_corner
            else:
                return False
        
        node_min = self.bounds[:3]
        node_max = self.bounds[3:]
        # 检查AABB重叠
        return not any(obj_max < node_min) and not (any(obj_min > node_max))
    
    def insert(self, obj):
        """将对象插入到八叉树中"""
        # 如果未分割且对象数量超过阈值，进行分割
        if not self.is_divided and len(self.objects) >= self.max_objects and self.depth < self.max_depth:
            self.subdivide()
        
        if self.is_divided:
            # 尝试插入子节点
            inserted = False
            for child in self.children:
                if child._aabb_intersects_object(obj):
                    child.insert(obj)
                    inserted = True
            # 如果未插入任何子节点，保留在当前节点
            if not inserted:
                self.objects.append(obj)
        else:
            # 直接添加到当前节点
            self.objects.append(obj)
    
    def query(self, obj):
        """查询可能与给定对象碰撞的候选对象"""
        candidates = list(self.objects)  # 当前节点的对象
        
        if self.is_divided:
            # 递归查询子节点
            for child in self.children:
                if child._aabb_intersects_object(obj):
                    candidates.extend(child.query(obj))
        
        return candidates


class Octree:
    """八叉树空间索引结构"""
    def __init__(self, objects, bounds=None, max_depth=4, max_objects=8):
        """
        初始化八叉树
        :param objects: 要索引的对象列表
        :param bounds: 自定义边界，如果为None则自动计算
        """
        if bounds is None:
            # 自动计算包围所有对象的边界
            min_corner = np.full(3, np.inf)
            max_corner = np.full(3, -np.inf)
            
            for obj in objects:
                if hasattr(obj, 'get_aabb'):
                    obj_min, obj_max = obj.get_aabb()
                else:
                    min_corner_obj, max_corner_obj = obj.get_bounding_box()
                    obj_min = min_corner_obj
                    obj_max = max_corner_obj
                
                min_corner = np.minimum(min_corner, obj_min)
                max_corner = np.maximum(max_corner, obj_max)
            
            # 扩展边界避免边缘问题
            size = max_corner - min_corner
            min_corner -= size * 0.1
            max_corner += size * 0.1
            bounds = np.concatenate([min_corner, max_corner])
        
        # 创建根节点
        self.root = OctreeNode(bounds, 0, max_depth, max_objects)
        
        # 插入所有对象
        for obj in objects:
            self.root.insert(obj)
    
    def query(self, obj):
        """执行空间查询，返回可能碰撞的对象"""
        return self.root.query(obj)


# ====================== 可达性分析器 ======================
class AccessibilityAnalyzer:
    """可达性分析器，评估工具对目标点的可达性"""
    def __init__(self, static_objects, tool_model):
        """
        初始化分析器
        :param static_objects: 静态障碍物列表
        :param tool_model: 工具模型
        """
        self.static_objects = static_objects
        self.tool_model = tool_model
        # 构建八叉树空间索引
        self.octree = Octree(static_objects)
        # 创建静态物体索引
        self.static_objects_dict = {id(obj): obj for obj in static_objects}
        
    def _calculate_collision_penalty(self, collision_count, total_checks):
        """
        基于碰撞检测结果的惩罚计算
        使用非线性响应函数和环境复杂度补偿因子
        :param collision_count: 实际碰撞次数
        :param total_checks: 总检测次数
        :return: 惩罚值 [0,1]
        """
        if total_checks == 0:
            return 0.0
            
        # 计算基础碰撞率
        collision_rate = collision_count / total_checks
        
        # 非线性惩罚响应函数 (S型曲线)
        if collision_rate <= 0.05:  # 低碰撞率区: 惩罚加重
            response_factor = 1.2
        elif collision_rate <= 0.20:  # 线性过渡区
            response_factor = 1.2 - (collision_rate - 0.05) * (0.7/0.15)
        else:  # 高碰撞率区: 惩罚减轻
            response_factor = 0.5
            
        # 环境复杂度补偿因子 (对数函数)
        # 检测次数越多 → 环境越复杂 → 惩罚减轻
        complexity_factor = 1.0 - 0.4 * min(1.0, math.log(total_checks + 1) / 5)
        
        # 基础惩罚值 (考虑碰撞次数)
        base_penalty = min(0.5, collision_count * 0.1)
        
        # 综合惩罚计算
        penalty = base_penalty * response_factor * complexity_factor
        
        # 确保惩罚在合理范围内
        return min(0.6, max(0.0, penalty))

    def evaluate_target_point(self, target_position, tool_position, tool_rotation):
        """
        评估单个目标点的可达性
        :param target_position: 目标点位置
        :param tool_position: 工具位置
        :param tool_rotation: 工具旋转 (欧拉角)
        :return: 包含各项评分的字典
        """
        # 创建工具代理对象用于空间查询
        class ToolProxy:
            def __init__(self, position, rotation, model):
                self.position = position
                self.rotation = rotation
                self.model = model
            
            def get_aabb(self):
                return self.model.get_aabb(self.position, self.rotation)
        
        tool_proxy = ToolProxy(tool_position, tool_rotation, self.tool_model)
        # 使用八叉树查询可能碰撞的对象
        candidate_objects = self.octree.query(tool_proxy)
        total_checks = len(candidate_objects)
        
        # 确保每个采样点至少有10次碰撞检测
        if total_checks < 10:
            needed = 10 - total_checks
            # 从静态物体中随机选择补充
            additional_objects = random.sample(self.static_objects, min(needed, len(self.static_objects)))
            
            # 避免重复添加
            candidate_ids = {id(obj) for obj in candidate_objects}
            for obj in additional_objects:
                if id(obj) not in candidate_ids:
                    candidate_objects.append(obj)
                    candidate_ids.add(id(obj))
            
            total_checks = len(candidate_objects)
        
        # 执行碰撞检测
        collision_count = 0
        collision_details = []
        for obj in candidate_objects:
            if self._check_tool_object_collision(tool_position, tool_rotation, obj):
                collision_count += 1
                collision_details.append(type(obj).__name__)
                
        # 计算碰撞惩罚
        collision_penalty = self._calculate_collision_penalty(collision_count, total_checks)

        # 计算工具头部中心位置
        head_local = np.array([-0.015, 0, 0])
        rotation = Rotation.from_euler('xyz', tool_rotation, degrees=True)
        head_center = tool_position + rotation.apply(head_local)
        
        # 距离评分: 工具头部到目标的距离
        dist = np.linalg.norm(target_position - head_center)
        max_dist = 0.3  # 最大有效距离
        if dist <= max_dist:
            dist_score = 1.0 - (dist / max_dist) * 0.5  # 距离越近评分越高
        elif dist <= max_dist * 1.5:
            dist_score = 0.5 * (1 - (dist - max_dist) / (max_dist * 0.5))
        else:
            dist_score = 0  # 超出有效范围

        # 角度评分: 工具方向与目标方向的夹角
        tool_dir = rotation.apply(np.array([1, 0, 0]))  # 工具朝向
        target_vec = target_position - tool_position
        if np.linalg.norm(target_vec) < 1e-6:
            angle_score = 0
            angle_deg = 180
        else:
            target_vec_normalized = target_vec / np.linalg.norm(target_vec)
            dot_product = np.dot(tool_dir, target_vec_normalized)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # 防止数值误差
            angle = np.arccos(dot_product)  # 计算夹角
            angle_deg = np.rad2deg(angle)
            angle_score = 1 - (angle / np.pi)  # 角度越小评分越高

        # 综合评分 = (距离分*0.6 + 角度分*0.4) * (1 - 碰撞惩罚)
        score = (dist_score * 0.6 + angle_score * 0.4) * (1 - collision_penalty)
        final_score = max(0.0, min(1.0, score))  # 确保在[0,1]范围内
        
        # 返回详细评估结果
        return {
            "final_score": final_score,
            "dist_score": dist_score,
            "angle_score": angle_score,
            "collision_count": collision_count,
            "collision_details": collision_details,
            "total_checks": total_checks,
            "collision_rate": collision_count / total_checks if total_checks > 0 else 0,
            "collision_penalty": collision_penalty,
            "distance": dist,
            "angle_deg": angle_deg
        }

    def _check_tool_object_collision(self, tool_position, tool_rotation, obj):
        """
        四阶段碰撞检测:
        1. 球体-球体快速检测
        2. OBB相交检测
        3. 精确网格碰撞检测
        """
        # 阶段1: 球体-球体检测 (快速排除)
        if not self._sphere_sphere_collision(tool_position, tool_rotation, obj):
            return False
        # 阶段2: OBB相交检测
        if not self._obb_collision(tool_position, tool_rotation, obj):
            return False
        # 阶段3: 精确网格碰撞检测
        return self._precise_mesh_collision(tool_position, tool_rotation, obj)
    
    def _sphere_sphere_collision(self, tool_position, tool_rotation, obj):
        """球体-球体碰撞检测 (快速排除)"""
        # 工具包围球
        tool_sphere_center = tool_position
        tool_sphere_radius = self.tool_model.handle_length * 0.8
        
        # 物体包围球
        if isinstance(obj, BoxModel):
            obj_sphere_center = obj.position
            obj_sphere_radius = np.linalg.norm(obj.size) / 2
        elif isinstance(obj, CylinderModel):
            obj_sphere_center = obj.position + np.array([0, 0, obj.height/2])
            obj_sphere_radius = max(obj.radius, obj.height/2)
        elif isinstance(obj, HumanModel):
            obj_sphere_center = obj.position + np.array([0, 0, obj.size[2]/2])
            obj_sphere_radius = np.linalg.norm(obj.size) / 2
        else:
            return False
        
        # 计算球心距离
        distance = np.linalg.norm(tool_sphere_center - obj_sphere_center)
        return distance <= (tool_sphere_radius + obj_sphere_radius)
    
    def _obb_collision(self, tool_position, tool_rotation, obj):
        """有向包围盒(OBB)碰撞检测"""
        tool_obb = self.tool_model.get_obb(tool_position, tool_rotation)
        obj_obb = obj.get_obb()
        return self._sat_obb_collision(tool_obb, obj_obb)
    
    def _sat_obb_collision(self, obb1, obb2):
        """分离轴定理(SAT)实现OBB碰撞检测"""
        T1, E1 = obb1  # 变换矩阵和半长
        T2, E2 = obb2
        
        R1 = T1[:3, :3]  # 旋转矩阵
        R2 = T2[:3, :3]
        t1 = T1[:3, 3]   # 平移向量
        t2 = T2[:3, 3]
        t = t2 - t1      # 相对平移
        R = R1.T @ R2    # 相对旋转
        
        # 检查obb1的轴
        for i in range(3):
            L = R1[:, i]  # 当前轴
            r1 = E1[i]    # obb1在该轴上的投影半径
            r2 = np.sum(E2 * np.abs(R[i, :]))  # obb2在该轴上的投影半径
            proj_dist = np.abs(np.dot(t, L))   # 平移在该轴上的投影
            if proj_dist > r1 + r2:  # 分离轴条件
                return False
        
        # 检查obb2的轴
        for i in range(3):
            L = R2[:, i]
            r1 = np.sum(E1 * np.abs(R[:, i]))
            r2 = E2[i]
            proj_dist = np.abs(np.dot(t, L))
            if proj_dist > r1 + r2:
                return False
        
        # 检查叉积轴 (15个轴)
        for i in range(3):
            for j in range(3):
                L = np.cross(R1[:, i], R2[:, j])
                if np.linalg.norm(L) < 1e-6:  # 忽略零向量
                    continue
                L = L / np.linalg.norm(L)  # 归一化
                r1 = np.sum(E1 * np.abs([np.dot(L, R1[:, k]) for k in range(3)]))
                r2 = np.sum(E2 * np.abs([np.dot(L, R2[:, k]) for k in range(3)]))
                proj_dist = np.abs(np.dot(t, L))
                if proj_dist > r1 + r2:
                    return False
        
        return True  # 所有轴都重叠，发生碰撞
    
    def _precise_mesh_collision(self, tool_position, tool_rotation, obj):
        """精确网格碰撞检测"""
        # 获取变换后的工具网格
        transformed_tool = self.tool_model.transform(tool_position, tool_rotation)
        
        # 获取物体网格
        if isinstance(obj, BoxModel):
            obj_mesh = trimesh.creation.box(obj.size)
            obj_mesh.apply_translation(obj.position)
        elif isinstance(obj, CylinderModel):
            obj_mesh = trimesh.creation.cylinder(radius=obj.radius, height=obj.height)
            obj_mesh.apply_translation(obj.position)
        elif isinstance(obj, HumanModel):
            obj_mesh = obj.get_mesh()
            obj_mesh.apply_translation(obj.position)
        else:
            return False

        # 使用trimesh的碰撞管理器进行精确检测
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object('tool', transformed_tool)
        collision_manager.add_object('object', obj_mesh)
        return collision_manager.in_collision_internal()


# ====================== 设备模型创建函数 ======================
def create_simple_equipment_model():
    """创建简易设备模型"""
    return BoxModel([0, 0, 0], [2, 1, 1.2], color=(0, 0.5, 0.8))


# ====================== 采样点生成函数 ======================
def generate_points_on_cylinder(center, height, radius, n_points=30):
    """
    在圆柱体表面生成采样点
    :param center: 圆柱中心
    :param height: 圆柱高度
    :param radius: 圆柱半径
    :param n_points: 总采样点数
    :return: 采样点列表
    """
    points = []
    # 分配点数: 20%用于顶底, 80%用于侧面
    top_bottom_count = int(n_points * 0.2)
    side_count = n_points - 2 * top_bottom_count

    # 在顶面和底面生成点
    for z_offset in [-height / 2, height / 2]:
        z = center[2] + z_offset
        for _ in range(top_bottom_count):
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, radius)  # 随机半径
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            points.append([x, y, z])

    # 在侧面生成点
    for _ in range(side_count):
        angle = random.uniform(0, 2 * np.pi)
        z = random.uniform(center[2] - height / 2, center[2] + height / 2)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append([x, y, z])

    return points


# ====================== 零件可达性评估函数 ======================
def evaluate_part_accessibility(part, analyzer, human_model, n_samples=150):
    """
    评估整个零件的可达性
    :param part: 目标零件
    :param analyzer: 可达性分析器
    :param human_model: 人体模型
    :param n_samples: 采样点数
    :return: 包含综合评估结果的字典
    """
    # 在零件表面生成采样点
    sample_points = generate_points_on_cylinder(
        part.position, part.height, part.radius, n_samples)

    # 基础工具位置 (基于人体肩部高度)
    base_tool_position = np.array([
        human_model.position[0],
        human_model.position[1],
        human_model.shoulder_height
    ])

    # 初始化结果收集列表
    all_scores = []  # 最终得分
    dist_scores = []  # 距离得分
    angle_scores = []  # 角度得分
    collision_counts = []  # 碰撞次数
    collision_details = []  # 碰撞详情
    total_checks_list = []  # 总检测次数
    collision_rates = []  # 碰撞率
    penalties = []  # 惩罚值

    # 对每个采样点进行评估
    for point in sample_points:
        # 添加随机变化使评估更真实
        angle_deviation = random.uniform(-45, 45)  # 角度偏差
        reach_variation = random.uniform(0.6, 1.4)  # 可达距离变化
        
        # 随机化工具位置
        tool_x = base_tool_position[0] + random.uniform(-0.2, 0.2)
        tool_y = base_tool_position[1] + random.uniform(0.2, 0.8) * reach_variation
        tool_z = base_tool_position[2] + random.uniform(-0.2, 0.2)
        tool_position = np.array([tool_x, tool_y, tool_z])

        # 计算指向目标的方向
        to_target = point - tool_position
        if np.linalg.norm(to_target) < 1e-6:
            yaw = 0
            pitch = 0
        else:
            to_target_normalized = to_target / np.linalg.norm(to_target)
            # 计算偏航角 (水平方向)
            yaw = np.arctan2(to_target_normalized[1], to_target_normalized[0])
            # 计算俯仰角 (垂直方向)
            pitch = np.arctan2(to_target_normalized[2], np.linalg.norm(to_target_normalized[:2]))

        # 构建工具旋转 (添加随机变化)
        tool_rotation = [
            np.rad2deg(pitch) + random.uniform(-30, 30),  # 俯仰
            np.rad2deg(yaw) + angle_deviation,             # 偏航
            random.uniform(-15, 15)                       # 滚转
        ]

        # 评估当前配置
        result = analyzer.evaluate_target_point(point, tool_position, tool_rotation)
        # 收集结果
        all_scores.append(result["final_score"])
        dist_scores.append(result["dist_score"])
        angle_scores.append(result["angle_score"])
        collision_counts.append(result["collision_count"])
        total_checks_list.append(result["total_checks"])
        collision_rates.append(result["collision_rate"])
        penalties.append(result["collision_penalty"])
        collision_details.extend(result["collision_details"])

    # 处理无有效点的情况
    if not all_scores:
        return {
            "final_score": 0.0,
            "avg_dist_score": 0.0,
            "avg_angle_score": 0.0,
            "collision_prob": 0.0,
            "severe_collision_ratio": 0.0,
            "max_collision": 0,
            "penalty_intensity": 0.0,
            "high_penalty_ratio": 0.0,
            "max_penalty": 0.0,
            "penalty_variation": 0.0,
            "collision_details": []
        }

    # 筛选有效点 (得分>0.05)
    valid_indices = [i for i, s in enumerate(all_scores) if s > 0.05]
    if not valid_indices:
        return {
            "final_score": 0.0,
            "avg_dist_score": 0.0,
            "avg_angle_score": 0.0,
            "collision_prob": sum(collision_rates) / len(collision_rates),
            "severe_collision_ratio": sum(1 for c in collision_counts if c >= 2) / len(collision_counts),
            "max_collision": max(collision_counts) if collision_counts else 0,
            "penalty_intensity": sum(penalties) / len(penalties),
            "high_penalty_ratio": sum(1 for p in penalties if p > 0.3) / len(penalties),
            "max_penalty": max(penalties) if penalties else 0,
            "penalty_variation": 0.0,
            "collision_details": collision_details
        }
    
    # 最终得分计算（加权平均，高分点权重更大）
    weights = [all_scores[i] ** 2 for i in valid_indices]
    weighted_sum = sum(all_scores[i] * w for i, w in zip(valid_indices, weights))
    final_score = weighted_sum / sum(weights)
    
    # 其他指标计算（算术平均）
    avg_dist_score = sum(dist_scores[i] for i in valid_indices) / len(valid_indices)
    avg_angle_score = sum(angle_scores[i] for i in valid_indices) / len(valid_indices)
    avg_total_checks = sum(total_checks_list[i] for i in valid_indices) / len(valid_indices)
    
    # 碰撞指标计算
    valid_collision_counts = [collision_counts[i] for i in valid_indices]
    collision_prob = sum(1 for c in valid_collision_counts if c > 0) / len(valid_collision_counts)
    severe_collision_ratio = sum(1 for c in valid_collision_counts if c >= 2) / len(valid_collision_counts)
    max_collision = max(valid_collision_counts)
    
    # 惩罚指标计算
    valid_penalties = [penalties[i] for i in valid_indices]
    penalty_intensity = sum(valid_penalties) / len(valid_penalties)
    high_penalty_ratio = sum(1 for p in valid_penalties if p > 0.3) / len(valid_penalties)
    max_penalty = max(valid_penalties)
    
    # 惩罚变异系数 (衡量惩罚值的波动性)
    if penalty_intensity > 1e-5:
        penalty_std = np.std(valid_penalties)
        penalty_variation = penalty_std / penalty_intensity
    else:
        penalty_variation = 0.0
    
    return {
        "final_score": final_score,
        "avg_dist_score": avg_dist_score,
        "avg_angle_score": avg_angle_score,
        "collision_prob": collision_prob,
        "severe_collision_ratio": severe_collision_ratio,
        "max_collision": max_collision,
        "penalty_intensity": penalty_intensity,
        "high_penalty_ratio": high_penalty_ratio,
        "max_penalty": max_penalty,
        "penalty_variation": penalty_variation,
        "avg_total_checks": avg_total_checks,
        "collision_details": collision_details
    }


# ====================== 可达性等级评估函数 ======================
def get_accessibility_level(score):
    """
    根据评分确定可达性等级和描述
    :param score: 归一化后的可达性评分 [0,1]
    :return: (等级, 描述)
    """
    if score <= 0.10:
        accessibility_level = "不可达"
        description = "无法接触"
    elif 0.10 < score <= 0.20:
        accessibility_level = "严重受限"
        description = "极难接触"
    elif 0.20 < score <= 0.30:
        accessibility_level = "高度受限"
        description = "接近受限"
    elif 0.30 < score <= 0.40:
        accessibility_level = "中度受限"
        description = "部分可操作"
    elif 0.40 < score <= 0.50:
        accessibility_level = "轻度受限"
        description = "可接触但不流畅"
    elif 0.50 < score <= 0.60:
        accessibility_level = "基本可达"
        description = "简单操作可行"
    elif 0.60 < score <= 0.70:
        accessibility_level = "良好可达"
        description = "可正常操作"
    elif 0.70 < score <= 0.80:
        accessibility_level = "高度可达"
        description = "操作流畅"
    elif 0.80 < score <= 0.90:
        accessibility_level = "极佳可达"
        description = "直接接触"
    else:
        accessibility_level = "完全可达"
        description = "无障碍接触"

    return accessibility_level, description


# ====================== 归一化函数 ======================
def normalize_scores(scores):
    """
    将评分归一化到[0,1]范围
    :param scores: 原始评分列表
    :return: 归一化后的评分列表
    """
    if not scores:
        return []
        
    min_val = min(scores)
    max_val = max(scores)
    
    if max_val - min_val < 1e-6:
        return [0.5] * len(scores)  # 所有值相同的情况
        
    return [(s - min_val) / (max_val - min_val) for s in scores]


# ====================== 主程序 ======================
if __name__ == "__main__":
    print("构建设备模型...")
    equipment = create_simple_equipment_model()
    
    # 人体特征参数 (5种不同体型)
    human_params = [
        {"position": [0.5, -0.3, 0], "size": [0.45, 0.25, 1.75]},
        {"position": [0.5, -0.4, 0], "size": [0.40, 0.20, 1.65]},
        {"position": [0.5, -0.2, 0], "size": [0.50, 0.30, 1.85]},
        {"position": [0.7, -0.3, 0], "size": [0.45, 0.25, 1.75]},
        {"position": [0.3, -0.3, 0], "size": [0.40, 0.20, 1.65]}
    ]
    
    # 工具参数 (5种不同尺寸的扳手)
    tool_params = [
        {"handle_length": 0.4, "head_diameter": 0.05},
        {"handle_length": 0.5, "head_diameter": 0.06},
        {"handle_length": 0.6, "head_diameter": 0.07},
        {"handle_length": 0.7, "head_diameter": 0.08},
        {"handle_length": 0.8, "head_diameter": 0.04}
    ]
    
    # 维修零件参数 (5个不同位置的零件)
    part_params = [
        {"position": [0, 0, 0.8], "height": 0.3, "radius": 0.1},
        {"position": [0.2, 0.1, 1.0], "height": 0.25, "radius": 0.08},
        {"position": [-0.1, -0.1, 0.9], "height": 0.35, "radius": 0.12},
        {"position": [0.3, 0.0, 1.1], "height": 0.2, "radius": 0.09},
        {"position": [-0.2, 0.15, 1.0], "height": 0.4, "radius": 0.15}
    ]
    
    results = []  # 存储所有评估结果
    
    # 遍历所有参数组合 (5人体×5工具×5零件=125种组合)
    for i, hp in enumerate(human_params):
        for j, tp in enumerate(tool_params):
            for k, pp in enumerate(part_params):
                print(f"\n评估组合 {i+1}-{j+1}-{k+1}...")
                print(f"人体参数: 位置={hp['position']}, 尺寸={hp['size']}")
                print(f"工具参数: 手柄={tp['handle_length']:.2f}m, 头部={tp['head_diameter']:.2f}m")
                print(f"零件参数: 位置={pp['position']}, 高度={pp['height']:.2f}m, 半径={pp['radius']:.2f}m")
                
                # 创建模型实例
                human = HumanModel(position=hp["position"], size=hp["size"])
                tool = WrenchToolModel(handle_length=tp["handle_length"], head_diameter=tp["head_diameter"])
                part = CylinderModel(position=pp["position"], height=pp["height"], radius=pp["radius"])
                
                # 静态物体列表 (设备和人体)
                static_objects = [equipment, human]
                # 初始化可达性分析器
                analyzer = AccessibilityAnalyzer(static_objects, tool)
                
                # 评估零件可达性 (150个采样点)
                evaluation_result = evaluate_part_accessibility(part, analyzer, human, n_samples=150)
                
                # 存储结果
                results.append({
                    "human_id": i+1,
                    "human_position": hp["position"],
                    "human_size": hp["size"],
                    "tool_id": j+1,
                    "tool_handle_length": tp["handle_length"],
                    "tool_head_diameter": tp["head_diameter"],
                    "part_id": k+1,
                    "part_position": pp["position"],
                    "part_height": pp["height"],
                    "part_radius": pp["radius"],
                    "raw_score": evaluation_result["final_score"],
                    "avg_dist_score": evaluation_result["avg_dist_score"],
                    "avg_angle_score": evaluation_result["avg_angle_score"],
                    "collision_prob": evaluation_result["collision_prob"],
                    "severe_collision_ratio": evaluation_result["severe_collision_ratio"],
                    "max_collision": evaluation_result["max_collision"],
                    "penalty_intensity": evaluation_result["penalty_intensity"],
                    "high_penalty_ratio": evaluation_result["high_penalty_ratio"],
                    "max_penalty": evaluation_result["max_penalty"],
                    "penalty_variation": evaluation_result["penalty_variation"],
                    "avg_total_checks": evaluation_result["avg_total_checks"],
                    "collision_details": evaluation_result["collision_details"]
                })
    
    # 归一化原始评分
    raw_scores = [res['raw_score'] for res in results]
    normalized_scores = normalize_scores(raw_scores)
    
    # 添加归一化评分和可达性等级
    for i, res in enumerate(results):
        normalized_score = normalized_scores[i]
        level, description = get_accessibility_level(normalized_score)
        res['normalized_score'] = normalized_score
        res['level'] = level
        res['description'] = description
    
    # 输出结果表格
    print("\n\n" + "="*150)
    print("实体可达性评估结果汇总")
    print("="*150)
    print(f"{'组合':<8} | {'人体ID':<6} | {'工具ID':<6} | {'零件ID':<6} | {'原始评分':<8} | {'归一化评分':<8} | {'距离评分':<8} | {'角度评分':<8} | "
          f"{'碰撞概率':<8} | {'严重碰撞比':<8} | {'最大碰撞':<6} | "
          f"{'惩罚强度':<8} | {'高惩罚比例':<8} | {'最大惩罚':<8} | {'惩罚变异':<8} | {'检测总数':<8} | {'可达性等级':<10} | {'操作状态说明'}")
    print("-"*150)
    
    for idx, res in enumerate(results):
        print(f"{idx+1:<8} | {res['human_id']:<6} | {res['tool_id']:<6} | {res['part_id']:<6} | "
              f"{res['raw_score']:.4f} | {res['normalized_score']:.4f} | "
              f"{res['avg_dist_score']:.4f} | {res['avg_angle_score']:.4f} | "
              f"{res['collision_prob']:.4f} | {res['severe_collision_ratio']:.4f} | {res['max_collision']:<6} | "
              f"{res['penalty_intensity']:.4f} | {res['high_penalty_ratio']:.4f} | {res['max_penalty']:.4f} | {res['penalty_variation']:.4f} | "
              f"{res['avg_total_checks']:.1f} | "
              f"{res['level']:<10} | {res['description']}")
    
    # 统计指标
    min_raw_score = min(raw_scores)
    max_raw_score = max(raw_scores)
    avg_raw_score = sum(raw_scores) / len(raw_scores)
    min_norm_score = min(normalized_scores)
    max_norm_score = max(normalized_scores)
    avg_norm_score = sum(normalized_scores) / len(normalized_scores)
    
    # 碰撞指标统计
    collision_probs = [res['collision_prob'] for res in results]
    severe_ratios = [res['severe_collision_ratio'] for res in results]
    max_collisions = [res['max_collision'] for res in results]
    
    # 惩罚指标统计
    penalty_intensities = [res['penalty_intensity'] for res in results]
    high_penalty_ratios = [res['high_penalty_ratio'] for res in results]
    max_penalties = [res['max_penalty'] for res in results]
    penalty_variations = [res['penalty_variation'] for res in results]
    
    # 输出统计摘要
    print("\n" + "="*150)
    print("评分统计:")
    print(f"- 最低原始评分: {min_raw_score:.4f}")
    print(f"- 最高原始评分: {max_raw_score:.4f}")
    print(f"- 平均原始评分: {avg_raw_score:.4f}")
    print(f"- 最低归一化评分: {min_norm_score:.4f}")
    print(f"- 最高归一化评分: {max_norm_score:.4f}")
    print(f"- 平均归一化评分: {avg_norm_score:.4f}")
    print("\n碰撞指标统计:")
    print(f"- 平均碰撞概率: {sum(collision_probs)/len(collision_probs):.4f}")
    print(f"- 平均严重碰撞比例: {sum(severe_ratios)/len(severe_ratios):.4f}")
    print(f"- 平均最大碰撞次数: {sum(max_collisions)/len(max_collisions):.1f}")
    print("\n惩罚指标统计:")
    print(f"- 平均惩罚强度: {sum(penalty_intensities)/len(penalty_intensities):.4f}")
    print(f"- 平均高惩罚比例: {sum(high_penalty_ratios)/len(high_penalty_ratios):.4f}")
    print(f"- 平均最大惩罚值: {sum(max_penalties)/len(max_penalties):.4f}")
    print(f"- 平均惩罚变异系数: {sum(penalty_variations)/len(penalty_variations):.4f}")
    print("="*150)
    
    # 尝试将结果导出到Excel
    try:
        df = pd.DataFrame(results)
        # 添加组合ID列
        df.insert(0, '组合ID', range(1, len(df)+1))
        # 调整列顺序
        df = df[['组合ID', 'human_id', 'human_position', 'human_size', 
                 'tool_id', 'tool_handle_length', 'tool_head_diameter', 
                 'part_id', 'part_position', 'part_height', 'part_radius', 
                 'raw_score', 'normalized_score', 
                 'avg_dist_score', 'avg_angle_score', 
                 'collision_prob', 'severe_collision_ratio', 'max_collision',
                 'penalty_intensity', 'high_penalty_ratio', 'max_penalty', 'penalty_variation',
                 'avg_total_checks',
                 'level', 'description', 'collision_details']]
        # 重命名列
        df.rename(columns={
            'human_id': '人体ID',
            'human_position': '人体位置',
            'human_size': '人体尺寸(m)',
            'tool_id': '工具ID',
            'tool_handle_length': '工具手柄长度(m)',
            'tool_head_diameter': '工具头部直径(m)',
            'part_id': '零件ID',
            'part_position': '零件位置',
            'part_height': '零件高度(m)',
            'part_radius': '零件半径(m)',
            'raw_score': '原始评分',
            'normalized_score': '归一化评分',
            'avg_dist_score': '平均距离评分',
            'avg_angle_score': '平均角度评分',
            'collision_prob': '碰撞概率',
            'severe_collision_ratio': '严重碰撞比例',
            'max_collision': '最大碰撞次数',
            'penalty_intensity': '惩罚强度',
            'high_penalty_ratio': '高惩罚比例',
            'max_penalty': '最大惩罚值',
            'penalty_variation': '惩罚变异系数',
            'avg_total_checks': '平均检测次数',
            'level': '可达性等级',
            'description': '操作状态说明',
            'collision_details': '碰撞详情'
        }, inplace=True)

        # 保存到Excel文件
        excel_filename = "实体可达性评估结果.xlsx"
        df.to_excel(excel_filename, index=False)
        print(f"\n评估结果已保存到: {excel_filename}")
    except Exception as e:
        print(f"\n生成Excel文件时出错: {str(e)}")
        print("请确保已安装pandas和openpyxl库")
        print("安装命令: pip install pandas openpyxl")
    
    print("\n评估完成")