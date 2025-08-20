import os
from comfort_cal import Comfort

def main():
    # 读取关节位置, 包括 当前的负重{load:kg}，比如搬运一个1kg的箱子,关节位置{joint:[x,y,z]}
    joints_json = os.path.join(os.path.dirname(__file__),"joints.json")

    # 调用 Comfort 类
    comfort = Comfort(joints_json)

    # 计算舒适度
    print(f"舒适度为:{comfort()}")


if __name__ == "__main__":
    main()
