# 全连接神经网络

### 应用

##### 手写数字识别0～9

1. 使用mnist训练集
    - 训练集为50000张图片
    - 测试集为10000张图片
    
2. 达到的效果
   拟合度99.70%,测试集上 **准确率98.38%** ，在训练了28个epoth获得



### 算法细节

1. 使用反向传播算法计算梯度dw和db
   

   <img width="249" alt="4个基本方程" src="https://user-images.githubusercontent.com/19931702/110571270-46533a80-8192-11eb-8fc0-6975d421bae0.png">


2. 采用L2正则化
3. 采用随机梯度下降算法
    - min-batch大小为10
4. 使用fmincg高级优化算法执行梯度下降的单次迭代
5. 数据归一化
6. 参数初始化
    - 生成第l层的w和b，k为第l-1层的神经元个数
    - 使用方差为 1/k 的高斯分布生成w，方差为1 的高斯分布生成b
    - 这一定程度上加快了训练速度，详细联系作者
7. 对输出层采用softmax层，得到每个类别概率分布的输出







### 更多测试细节

**不同超参数得到的训练结果报告**

详细数据报告联系作者

| 超参数 | 训练集上拟合度    |  测试集上准确率  | 训练的epoth数 |
| :--------   | -----:   | :----:  | :----: |
| 未优化的bp神经网络        |      |     |        |  
| 优化算法改变 、将fminunc 改成fmincg        |     |     |        |
| 参数随机初始化，使用方差为 1/k 的高斯分布生成w，方差为1 的高斯分布生成b        |  |      |        |
| batchGD 改成 随机梯度下降        |  |      |        |
| 数据归一化        |  |      |        |
| [784 8 10] 修改为 [784 10 10]        |  |      |        |
| [784 10 10] 修改为 [784 30 10], min-batch itertaion 1 增加到2        |  |      |        |
|  [784 30 10], min-batch itertaion 2 修改为1        |  |      |        |
| 修改 cost 函数  为 分段计算 ;以消除0*inf=NaN问题。       |  |      |        |
| [784 30 10] 修改为[784 40 10]    |99.01% |96.49%|39|
|10得到结果图，train和test 在cost 和accuracy上的差距随着训练次数增大而增大，存在一定的过度拟合，使用L2正则化,lambda=0.01,[784 40 10] 修改为[784 48 10],   |93.42% |93.79%|33|
|11得到结果图,表明了具有较高的泛化能力，但存在一定的欠拟合， [784 48 10] 修改为[784 64 10] |93.77% |94.25%|26|
|lambda=0.001                    |97.19% |96.79%|36|
|[784 64 10] 修改为[784 90 10]    |97.47% |96.94%|23|
|[784 90 10] 修改为[784 100 10]   |97.43% |97.13%|54|
|[784 100 10] 修改为[784 200 10]  |97.36% |97.22%|47|
|lambda=0.00005                  |99.05% |97.98%|18|
|lambda=0.000001                 |99.71% |98.08%|23|
|[784 200 10] 修改为[784 256 10]  |99.66% |98.23%|17|
|lambda=0.00001                  |99.48% |98.16%|22|
|[784 256 10] 修改为[784 300 10]  |99.61% |98.24%|40|
|lambda=0.0000001                |99.86% |98.32%|28|
|[784 300 10] 修改为[784 800 10]  |99.70% |98.38%|28|
|[784 800 10] 修改为[784 200 10] lambda=0.00000001  |99.93% |98.24%|50|
|添加 softmax 层                  |99.78% |98.27%|30|
 
 
    
    
