import os
import json
import numpy as np
from typing import List,Dict


def read_json(path:str)->Dict:
    try:
        with open(path,"r",encoding = "utf-8") as file:
            data = json.load(file)
            return data
    
    except Exception as e:
        raise(e)


# class Comfort():
#     '''
#     将 rula分值映射到量表
#     '''

#     rula2com_list = [(0,0),(2,1),(4,2),(6,3),(7,4)]
#     comfort_table = {0:"舒适性优秀，无风险",
#                     1:"舒适性良好，低风险，可接受，无需采取行动",
#                     2:"舒适性一般，中等风险，可能需要进行改进",
#                     3:"舒适性较差，较高风险，需尽快采取相应的措施进行人因改进",
#                     4:"舒适性很差，极高风险，需立即采取相应的措施进行人因改进"}

#     def __init__(self):
#         self.rulas = list()
#         self.comfort_list = list()

#     @property
#     def _rula2com(self,rula):
#         for r,c in rula2com_list:
#             if rula<=r:
#                 comfort_value = c
#                 break
        
#         warning = self.comfort_table[rula]

#         self.comfort_list.append(comfort_value)
#         assert len(self.rulas)==len(self.comfort_list)

#     def get_rula(self,rula):
#         assert 0<=rula<=7,"rula值无效"
#         self.rulas.append(rula)
#         comfort_value,warning = self._rula2com(rula)

#         return comfort_value,warning

#     def run_warning(self,rula):
#         comfort_value,warning = self.get_rula(rula)
#         if comfort_value>=3:
#             return warning


#     def reset(self):
#         self.rulas = list()
#         self.comfort_list = list()


class Comfort():

    table_path:str=os.path.join(os.path.dirname(__file__),"rula_table.json")

    # 这里的读取应 对应 真实读取的关节名称
    rula_joints: List[str] = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
     'right_wrist', 'left_wrist', 'right_middle1', 'left_middle1', 'neck', 'spine3', 'head', 'pelvis', 'spine1']
    child:List[str] = [['pelvis','spine1'],['spine1','spine3'],['spine3','neck'],['spine3','left_shoulder'],['spine3','right_shoulder'],
     ['left_shoulder','left_elbow'],['right_shoulder','right_elbow'],['left_elbow','left_wrist'],['right_elbow','right_wrist'],
     ['neck','head'],['left_wrist','left_middle1'],['right_wrist','right_middle1']]
    parents:Dict[str,str]={child:parent for child,parent in child}

    rula2com_list = [(0,0),(2,1),(4,2),(6,3),(7,4)]

    def __init__(self,info_path:str):
        '''
        初始化rula类
        :param info_path: 信息路径，为每帧存储的关节路径和负重信息，以json格式存储，类似 {"关节名称":[x,y,z]}， {"load":kg}

        '''
        self.joint_data = read_json(info_path) # 读取关节数据
        self.tables = read_json(self.table_path)
        self.arm_score = list()
        self.leg_score = list()
        self.loads = self.joint_data['load']
    
    def __call__(self) -> int:
        '''
        计算rula值
        :return: rula值
        '''
        rulas = list()

        for slid in ['left','right']:
            self.slid = slid

            # 得到两个分数后，查第三个表
            arm_score = self._get_arm_score()
            leg_score = self._get_leg_score()

            A_trans = lambda x, y: 3 * (x - 1) + y - 1
            B_trans = lambda x, y: 2 * (x - 1) + y - 1
            wrist_score = 1
            balanced_score,pose_scores = 1,0


            A_score = self.tables['table_A'][A_trans(arm_score[0], arm_score[1])][A_trans(arm_score[2], wrist_score)]
            B_score = self.tables['table_B'][leg_score[0]-1][B_trans(leg_score[1], balanced_score)]
            A_score = A_score if A_score < 9 else 7
            B_score = B_score if B_score < 8 else 6

            if self.loads < 2:
                loads_score = 0
            elif 2 <= self.loads <= 10:
                if pose_scores == 0:
                    loads_score = 1
                else:
                    loads_score = 2
            else:
                loads_score = 3

            rula_score = self.tables['table_C'][A_score + pose_scores + loads_score -1][B_score + pose_scores + loads_score -1]
            rulas.append(rula_score)
        rula_score = max(rulas)

        for r,c in self.rula2com_list:
            if rula_score<=r:
                comfort_value = c
                break

        return comfort_value

    def get_angle(self,joint_name:str)->float:
        '''
        从 dict 中读取关节位置，计算向量之间的夹角
        param:
        joint_names: a,b,c,计算 ab,ac之间的夹角
        joint_data:按照 名称，坐标进行存储
        '''
        joint_names:List[str] = [joint_name]

        joint_names.append(self.parents[joint_name])
        
        # 从数组中找到匹配的
        for p,c in self.child:
            if p == joint_name:
                joint_names.append(c)
                
        def cal_angel(vector1:np.array,vector2:np.array)->float:
            '''
            计算向量之间的夹角
            '''
            vector1 = vector1 if not isinstance(vector1,np.ndarray) else np.array(vector1)
            vector2 = vector2 if not isinstance(vector2,np.ndarray) else np.array(vector2)

            dot_product = np.dot(vector1,vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)

            cos_theta = dot_product/(norm1*norm2)
            theta = np.arccos(cos_theta)

            assert -np.pi<=theta<=np.pi

            return theta


        coordinates = list()
        for name in joint_names:
            v = self.joint_data[name]
            coordinates.append(np.array(v))

        vector1 = coordinates[1]-coordinates[0]
        vector2 = coordinates[2]-coordinates[0]

        theta = cal_angel(vector1,vector2)

        return theta


    def _get_arm_score(self)->List[int]:

        '''
        计算手臂分数，返回三个分数
        :return: [上臂,前臂,手腕 ]
        '''
        arms = ['_shoulder','_elbow','_wrist']
        scores = []
        for i in arms:
            if i == '_shoulder':
                angle = self.get_angle(self.slid + i)
                if angle < 20:
                    scores.append(1)
                elif 20<= angle < 45:
                    scores.append(2)
                elif 45<= angle < 95:
                    scores.append(3)
                else:
                    scores.append(4)

            if i == '_elbow':
                angle = self.get_angle(self.slid + i)
                angle = 180-angle
                if 60<=angle < 100:
                    scores.append(1)
                elif angle < 60:
                    scores.append(2)
                else:
                    scores.append(3)

            if i == '_wrist':
                angle = self.get_angle(self.slid+i)
                angle = 180-angle
                if angle < 5:
                    scores.append(1)
                elif 5<= angle < 15:
                    scores.append(2)
                else:
                    scores.append(3)

        return scores

    def _get_leg_score(self)->List[int]:
        '''返回腿部分数'''
        joints = ['neck','trunk']
        scores = []
        for i in joints:
            if i == 'neck':
                angle = self.get_angle(i)
                if angle < 10:
                    scores.append(1)
                elif 10<= angle < 20:
                    scores.append(2)
                else:
                    scores.append(3)

            if i == 'trunk':
                angle = self.get_angle('pelvis')
                if angle < 5:
                    scores.append(1)
                elif 5<= angle < 20:
                    scores.append(2)
                elif 20<= angle < 60:
                    scores.append(3)
                else:
                    scores.append(4)
        return scores


