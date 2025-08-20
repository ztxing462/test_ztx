import json
import os

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 时发生未知错误: {e}")


class Fatigue:
    energy_path = os.path.join(os.path.dirname(__file__), 'energy.json') # 字典，{动作名称：能耗值},单位：秒,以及 {'weight':kg}

    def __init__(self,action_json) -> None:
        '''
        读取动作序列文件，计算疲劳值
        动作序列应为 拆解的动作名称，时间
        '''
        self.info_json = read_json_file(action_json) # 字典，{动作名称：时间},单位：秒,以及 {'weight':kg}
        self.action_json = self.info_json['actions']
        self.weight = self.info_json['weight']
        self.action_list = self.action_json.keys()
        self.energy_dict = self.get_energy()  # 字典，{动作名称：能量值}

    def __call__(self):
        '''
        计算疲劳值
        '''
        total_energy,t_cnt = 0,0
        for action in self.action_list:
            if action == 'weight':
                continue
            action_time = self.action_json[action]
            t_cnt+=self.action_json[action]
            
            if action not in self.energy_table.keys():
                action = 'other'

            total_energy += self.weight * self.energy_dict[action] * action_time/60 *4.184 #kj  #这里应该是 3600，但和下面的秒进行合并了
        total_energy+=1.44 *t_cnt/60*4.184
        total_energy/=t_cnt #kj/min
        return total_energy
    
    def get_energy(self):
        '''
        读取能量文件
        '''
        self.energy_table = read_json_file(self.energy_path)
        energy_dict = {}

        # 检查动作序列中的动作是否在能量文件中
        for action in self.action_list:
            if action == 'weight':
                continue

            if action not in self.energy_table.keys():
                action = 'other'

            energy = self.energy_table[action]
            energy_dict[action] = energy
        
        return energy_dict




