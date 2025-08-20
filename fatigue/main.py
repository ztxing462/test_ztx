import os
from fatigue_cal import Fatigue
def main():
    '''example 展示'''

    # 动作序列文件,包含人体体重以及 整体拆解后每个动作的时间
    action_json = os.path.join(os.path.dirname(__file__), 'action_example.json')

    # 初始化疲劳类
    fatigue = Fatigue(action_json)

    # 调用疲劳类的__call__方法,计算疲劳值,该值可对应我们的疲劳度量表
    f = fatigue()
    print(f"疲劳值为：{f} kj/min")


if __name__ == "__main__":
    main()
