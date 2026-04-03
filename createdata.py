import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  #  Windows 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常

class SemiAnnulusRegion:
    """
    半环形区域类，用于生成区域内的随机点
    属性:
        r_inner: 内半径
        r_outer: 外半径
        center: 圆心坐标 (x, y)
        width: 环的宽度 (r_outer - r_inner)
        is_upper: 是否为上半环 (True=上半环, False=下半环)
    """
    def __init__(self, r_inner: float, r_outer: float, center: Tuple[float, float] = (0, 0), is_upper: bool = True):
        """
        初始化半环形区域
        :param r_inner: 内半径
        :param r_outer: 外半径
        :param center: 圆心坐标 (x, y)，默认原点
        :param is_upper: 是否为上半环，默认True
        """
        if r_inner >= r_outer:
            raise ValueError("内半径必须小于外半径")
        if r_inner < 0 or r_outer < 0:
            raise ValueError("半径不能为负数")
            
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.center_x, self.center_y = center
        self.is_upper = is_upper
        self.width = r_outer - r_inner  # 环的宽度属性

    def get_center(self) -> Tuple[float, float]:
        """获取圆心坐标"""
        return (self.center_x, self.center_y)

    def get_width(self) -> float:
        """获取环的宽度"""
        return self.width

    def get_radii(self) -> Tuple[float, float]:
        """获取内外半径"""
        return (self.r_inner, self.r_outer)

    def get_equidistant_points(self, n: int) -> np.ndarray:
        """
        生成半圆环【中圈】上的 n 等分坐标点
        专门用于：KMeans 固定初始聚类中心
        :param n: 等分点数
        :return: shape (n, 2) 的坐标点，均匀分布在半圆环中圈
        """
        # 用中圈半径（内外半径中间），保证点在环正中间
        mid_r = (self.r_inner + self.r_outer) / 2

        # 生成半圆 n 等分角度
        if self.is_upper:
            thetas = np.linspace(0, np.pi, n)  # 上半圆 0 → π
        else:
            thetas = np.linspace(np.pi, 2 * np.pi, n)  # 下半圆 π → 2π

        # 计算坐标
        x = self.center_x + mid_r * np.cos(thetas)
        y = self.center_y + mid_r * np.sin(thetas)

        # 拼成 (n, 2) 格式
        points = np.column_stack((x, y))
        return points

    def get(self, n: int, random_seed: int = 42) -> np.ndarray:
        """
        在区域内生成n个随机点
        :param n: 随机点数量
        :param random_seed: 随机种子，保证可复现
        :return: 形状为 (n, 2) 的数组，直接可传入 KMeans 训练
        """
        # 固定当前方法的随机种子
        rng = np.random.RandomState(random_seed)
        
        # 极坐标生成随机点
        r = np.sqrt(self.r_inner**2 + (self.r_outer**2 - self.r_inner**2) * rng.rand(n))
        
        if self.is_upper:
            theta = rng.uniform(0, np.pi, n)
        else:
            theta = rng.uniform(np.pi, 2 * np.pi, n)

        # 极坐标转直角坐标
        x = self.center_x + r * np.cos(theta)
        y = self.center_y + r * np.sin(theta)

        # 拼接成 (n, 2) 格式，直接用于 KMeans
        points = np.column_stack((x, y))
        return points

    def plot(self, color: str = None, label: str = None, alpha: float = 0.4):
        """
        绘制当前半环形区域（填充区域，无散点）
        :param color: 填充颜色
        :param label: 图例名称
        :param alpha: 透明度
        """
        # 自动配色
        if color is None:
            color = 'skyblue' if self.is_upper else 'lightgray'
        
        # 自动标签
        if label is None:
            label = "上半环" if self.is_upper else "下半环"

        # 生成角度
        theta = np.linspace(0, 2 * np.pi, 1000)
        if self.is_upper:
            theta_arc = theta[theta <= np.pi]
        else:
            theta_arc = theta[theta >= np.pi]

        # 内圈 + 外圈坐标
        x_inner = self.center_x + self.r_inner * np.cos(theta_arc)
        y_inner = self.center_y + self.r_inner * np.sin(theta_arc)
        x_outer = self.center_x + self.r_outer * np.cos(theta_arc)
        y_outer = self.center_y + self.r_outer * np.sin(theta_arc)

        # 拼接闭合路径（外圈顺向 + 内圈反向）
        x_fill = np.concatenate([x_outer, x_inner[::-1]])
        y_fill = np.concatenate([y_outer, y_inner[::-1]])

        # 填充区域
        plt.fill(x_fill, y_fill, color=color, alpha=alpha, label=label)

        # 绘制边界
        plt.plot(x_inner, y_inner, '--', color=color, alpha=0.8, linewidth=1.2)
        plt.plot(x_outer, y_outer, '--', color=color, alpha=0.8, linewidth=1.2)

        plt.axis('equal')
        plt.grid(alpha=0.3)

        plt.axhline(0, color='black', linewidth=0.8)
        plt.axvline(0, color='black', linewidth=0.8)


if __name__ == "__main__":

    for d in range(1,-7,-1):
        region1 = SemiAnnulusRegion(2, 3, center=(0, 0), is_upper=True)
        region2 = SemiAnnulusRegion(2, 3, center=(2.5, -d), is_upper=False)

        plt.figure(figsize=(7,7))
        region1.plot(color='dodgerblue', label='Region A')
        region2.plot(color='orange', label='Region B')
        
        plt.legend()
        plt.title("半环形区域填充图")

        # 保存图片
        filename = f"semi_annulus_d_{d}.png"  # 根据d值命名文件
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存图片: {filename}")
        
        plt.show()
