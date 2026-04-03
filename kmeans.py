import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"  # 加这行！

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  #  Windows 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常

class MyKMeans:
    def __init__(self, n_clusters: int, data: np.ndarray):
        """
        初始化 KMeans 聚类类
        :param n_clusters: 目标类别数
        :param data: 输入数据 (样本数, 特征数)
        """
        self.n_clusters = n_clusters  # 目标类别数
        self.data = data              # 输入数据
        self.model = None             # KMeans模型
        self.labels = None            # 聚类标签

    def fit(self, init_centers=None):
        """
        训练聚类
        :param init_centers: 手动指定初始中心，固定聚类位置
        """
        if init_centers is not None:
            self.model = KMeans(n_clusters=self.n_clusters, init=init_centers, n_init=1, random_state=42)
        else:
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = self.model.fit_predict(self.data)

    def get_means(self):
        """获取每个类别 各特征的均值（聚类中心）"""
        if self.labels is None:
            raise Exception("请先调用 fit() 训练模型")
        return self.model.cluster_centers_

    def get_variances(self):
        """获取每个类别 各特征的方差"""
        if self.labels is None:
            raise Exception("请先调用 fit() 训练模型")

        variances = []
        for k in range(self.n_clusters):
            # 取出第k类的所有样本
            cluster_data = self.data[self.labels == k]
            # 计算每个特征的方差
            var = np.var(cluster_data, axis=0)
            variances.append(var)
        return np.array(variances)
    
    def get_cluster_sizes(self):
        """
        获取每个类别的数据点数量（样本数）
        :return: list -> [类别0数量, 类别1数量, ..., 类别K-1数量]
        """
        if self.labels is None:
            raise Exception("请先调用 fit() 训练模型")
        
        cluster_sizes = []
        for k in range(self.n_clusters):
            # 统计第k类的样本点数
            count = np.sum(self.labels == k)
            cluster_sizes.append(count)
        return cluster_sizes

    def plot(self, title: str = "KMeans 聚类结果", feature_names: list = ["特征1", "特征2"]):
        """
        绘制聚类结果：原始点 + 聚类中心
        自动取前两维特征绘图
        """
        if self.labels is None:
            raise Exception("请先训练模型 fit()")

        # 取前两维特征
        X = self.data[:, 0]
        Y = self.data[:, 1]
        centers = self.get_means()
        cx = centers[:, 0]
        cy = centers[:, 1]

        plt.figure(figsize=(8, 6))

        # 绘制原始样本点（按类别上色）
        scatter = plt.scatter(
            X, Y,
            c=self.labels,       # 按聚类标签上色
            cmap="viridis",      # 配色
            s=30, alpha=0.6
        )

        # 绘制聚类中心（大红大星星，显眼）
        plt.scatter(
            cx, cy,
            c="red", marker="*",
            s=300, edgecolors="black", linewidth=2,
            label="聚类中心"
        )

        # 图例
        plt.legend(*scatter.legend_elements(), title="类别")
        plt.title(title, fontsize=14)
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
        plt.grid(alpha=0.3)
        plt.axis("equal")
        
        plt.show()


if __name__ == "__main__":

    # 1. 构造测试数据
    data = np.random.randn(300, 2)  # 200个样本，2个特征

    # 2. 创建聚类类
    kmeans = MyKMeans(n_clusters=3, data=data)

    # 3. 训练
    kmeans.fit()

    # 4. 获取均值（聚类中心）
    means = kmeans.get_means()
    print("各类别各特征均值：")
    print(means)

    # 5. 获取方差
    variances = kmeans.get_variances()
    print("\n各类别各特征方差：")
    print(variances)

    # 6. 画图
    kmeans.plot(title="半环区域 KMeans 聚类", feature_names=["X坐标", "Y坐标"])
    