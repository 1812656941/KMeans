from createdata import SemiAnnulusRegion
from kmeans import MyKMeans
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #  Windows 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常

def task1():
    K = 6
    answers = {}
    kmeans = MyKMeans(n_clusters=K, data=None)


    for d in range(1, -7, -1):
        # 创建半圆环区域
        region1 = SemiAnnulusRegion(2, 3, center=(0, 0), is_upper=True)
        region2 = SemiAnnulusRegion(2, 3, center=(2.5, -d), is_upper=False)

        # 随机采样数据点
        x1 = region1.get(2000, random_seed=42)
        x2 = region2.get(2000, random_seed=43)
        input = np.vstack((x1, x2))

        # 合成输入数据
        kmeans.data = input

        # 生成初始聚类中心
        upper_points = region1.get_equidistant_points(3)   # 上半圆 3 等分
        lower_points = region2.get_equidistant_points(3)  # 下半圆 3 等分
        fixed_init = np.vstack((upper_points, lower_points))  # 拼接成 6 个点

        # 开始聚类
        kmeans.fit(init_centers=fixed_init)

        # 绘制聚类结果
        kmeans.plot()

        # 保存聚类结果
        answers[d] = {
            'means': kmeans.get_means(),
            'variances': kmeans.get_variances(),
            'sizes': kmeans.get_cluster_sizes()
        }

        # 打印聚类结果
        print(answers[d])

    return answers
        

def task2(answers):
    # d 的顺序：1 → 0 → -1 → -2 → -3 → -4 → -5 → -6
    d_list = [1, 0, -1, -2, -3, -4, -5, -6]

    # 取出 1、2、3 类均值
    traj = np.array([answers[d]['means'][:3] for d in d_list])  # [8, 3, 2]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['第1类', '第2类', '第3类']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for cls_idx in range(3):
        head_width = 0.015
        head_length = 0.03
        if cls_idx == 2:
            head_width = 0.008
            head_length = 0.008
        ax = axes[cls_idx]
        x = traj[:, cls_idx, 0]
        y = traj[:, cls_idx, 1]

        # 1. 绘制轨迹线
        ax.plot(x, y, color=colors[cls_idx], linewidth=2, zorder=1)
        
        # 2. 绘制中间的点
        ax.scatter(x[1:-1], y[1:-1], color=colors[cls_idx], s=120, edgecolors='black', zorder=3)
        
        # 3. 绘制起点 (绿色)
        ax.scatter(x[0], y[0], color='lime', s=150, edgecolors='black', label='起点', zorder=4)
        
        # 4. 绘制终点 (红色)
        ax.scatter(x[-1], y[-1], color='red', s=150, edgecolors='black', label='终点', zorder=4)

        # 5. 绘制小箭头
        for j in range(len(x) - 1):
            dx = x[j+1] - x[j]
            dy = y[j+1] - y[j]
            ax.arrow(
                x[j], y[j], dx * 0.88, dy * 0.88,
                head_width=head_width, head_length=head_length,
                fc=colors[cls_idx], ec=colors[cls_idx],
                length_includes_head=True, zorder=2
            )

        ax.set_title(labels[cls_idx] + ' 均值变化轨迹', fontsize=20)
        ax.set_xlabel('X 坐标', fontsize=20)
        ax.set_ylabel('Y 坐标', fontsize=20)
        ax.tick_params(axis='x', labelsize=18)  # X轴刻度字体大小
        ax.tick_params(axis='y', labelsize=18)  # Y轴刻度字体大小
        ax.grid(alpha=0.3)
        ax.axis('equal')
        ax.legend()

    plt.tight_layout()
    plt.show()

def task3(answers):
    """
    绘制 answers 中方差变化图 → 风格和你的 KMeans plot 完全一致
    """
    d_list = sorted(answers.keys(), reverse=True)
    K = 6

    # 把方差读成数组 [d数量, K, 2]
    var_array = np.array([answers[d]['variances'] for d in d_list])

    colors = plt.cm.viridis(np.linspace(0, 1, K))
    cluster_labels = [f"类别 {i+1}" for i in range(K)]

    plt.figure(figsize=(10, 7))

    # ================= X 方差 ==================
    plt.subplot(2, 1, 1)
    for i in range(K):
        plt.plot(
            d_list, var_array[:, i, 0],
            marker='o', linewidth=2, markersize=7,
            color=colors[i], label=cluster_labels[i]
        )
    plt.title("各类别 X 方向方差随间距 d 的变化", fontsize=14)
    plt.ylabel("X 方差", fontsize=12)
    plt.grid(alpha=0.3)  # 同款网格
    plt.legend(title="类别")
    plt.tick_params(labelsize=11)

    # ================= Y 方差 ==================
    plt.subplot(2, 1, 2)
    for i in range(K):
        plt.plot(
            d_list, var_array[:, i, 1],
            marker='s', linewidth=2, markersize=7,
            color=colors[i], label=cluster_labels[i]
        )
    plt.title("各类别 Y 方向方差随间距 d 的变化", fontsize=14)
    plt.xlabel("间距 d", fontsize=12)
    plt.ylabel("Y 方差", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title="类别")
    plt.tick_params(labelsize=11)

    plt.tight_layout()
    plt.show()

def task4(answers):
    """
    绘制 X方差、Y方差 + 理论共同方差
    """
    d_list = sorted(answers.keys(), reverse=True)
    K = 6

    # 读取方差数据 [n_d, K, 2]
    var_array = np.array([answers[d]['variances'] for d in d_list])

    # ===================== 计算 第四问 理论共同方差 =====================
    common_var_theory = []
    for d in d_list:
        centers = answers[d]['means']  # 6个聚类中心
        d_max = 0
        # 计算所有中心两两之间的最大距离
        for i in range(K):
            for j in range(i + 1, K):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > d_max:
                    d_max = dist
        # 题目公式
        sigma = d_max / np.sqrt(2 * K)
        common_var_theory.append(sigma ** 2)
    print("common_var_theory：", common_var_theory)
    # ===================== 绘图 =====================
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    cluster_labels = [f"类别 {i+1}" for i in range(K)]

    plt.figure(figsize=(10, 7))

    # 子图1：X 方差 + 理论共同方差
    plt.subplot(2, 1, 1)
    for i in range(K):
        plt.plot(d_list, var_array[:, i, 0],
                 marker='o', linewidth=2, markersize=7,
                 color=colors[i], label=cluster_labels[i])
    # 理论共同方差（黑色粗线）
    plt.plot(d_list, common_var_theory, 'o-', color='black',
             linewidth=3, markersize=8, label="理论共同方差")
    plt.title("各类别 X方差 & 理论共同方差", fontsize=14)
    plt.ylabel("X 方差", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title="类别")
    plt.tick_params(labelsize=11)

    # 子图2：Y 方差 + 理论共同方差
    plt.subplot(2, 1, 2)
    for i in range(K):
        plt.plot(d_list, var_array[:, i, 1],
                 marker='s', linewidth=2, markersize=7,
                 color=colors[i], label=cluster_labels[i])
    # 理论共同方差（黑色粗线）
    plt.plot(d_list, common_var_theory, 'o-', color='black',
             linewidth=3, markersize=8, label="理论共同方差")
    plt.title("各类别 Y方差 & 理论共同方差", fontsize=14)
    plt.xlabel("间距 d", fontsize=12)
    plt.ylabel("Y 方差", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title="类别")
    plt.tick_params(labelsize=11)

    plt.tight_layout()
    plt.show()

def task5(answers):
    """
    绘制 6 个类别的样本点数（sizes）随间距 d 的变化曲线
    """
    # d 的顺序：1,0,-1,-2,-3,-4,-5,-6
    d_list = sorted(answers.keys(), reverse=True)
    K = 6

    # 读取 sizes 数据 [n_d, K]
    size_array = np.array([answers[d]['sizes'] for d in d_list])

    # 配色和你的其他图保持一致
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    cluster_labels = [f"类别 {i+1}" for i in range(K)]

    plt.figure(figsize=(10, 6))

    # 绘制每个类别的点数变化曲线
    for i in range(K):
        plt.plot(
            d_list, size_array[:, i],
            marker='o', linewidth=3, markersize=8,
            color=colors[i], label=cluster_labels[i]
        )

    # 图表样式（和你风格统一）
    plt.title("各类别样本点数量随间距 d 的变化", fontsize=16)
    plt.xlabel("间距 d", fontsize=14)
    plt.ylabel("类别样本点数", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(title="类别", fontsize=11)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    answers = task1()
    task2(answers)
    task3(answers)
    task4(answers)
    task5(answers)





