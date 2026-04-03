<img width="610" height="509" alt="image" src="https://github.com/user-attachments/assets/af7a9b74-498d-4733-8b50-6c0f40f58dc1" /># KMeans
这是一个KMeans算法在双月亮数据集上的实验代码 

This is an experimental code of the KMeans algorithm on the double moon dataset

实现半圆环区域类（SemiAnnulusRegion），该类属性包括内半径（r_inner）、外半径（r_outer）、环宽度（width）、圆心坐标（(x_center，y_center)）和是否为上半圆（is_upper），方法包括初始化、获取圆心坐标、获取环宽度、获取内外半径、获取随机点、获取等分点和绘制半圆环图。

MyKMeans是一个本实验对sklearn.cluster.KMeans进行封装的自定义类，提供函数接口来管理聚类过程、获取统计信息并进行可视化。类是属性包括目标聚类数和输入数据，方法包括拟合数据方法（fit）、获取聚类中心方法（get_means）、获取类内方差方法（get_variances）和绘制聚类结果方法。



