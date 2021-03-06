# 全连接神经网络

### 应用

##### 手写数字识别0～9

1. 使用[mnist训练集](http://neuralnetworksanddeeplearning.com/chap4.html)
    - 训练集为50000张图片
    - 测试集为10000张图片
    
2. 达到的效果
   - 拟合度99.70%,测试集上 **准确率98.38%** ，在训练了28个epoth获得



3. 预测实例

<img width="324" alt="" src="https://user-images.githubusercontent.com/19931702/110586281-eddd6680-81ac-11eb-80f3-fa8388a4fb58.png">

具体的，网络将会输出图片属于各个类别的概率向量，
如对于上面这张图片，
|类别  |概率 |
|---- | ---- |
|数字0| 1.1597e-13| 
|数字1| 1.1009e-11| 
|数字2| 1.0542e-10| 
|数字3| 7.5056e-14| 
|数字4| 1.0000e+00| 
|数字5| 1.3992e-12| 
|数字6| 2.2082e-13| 
|数字7| 2.9726e-09| 
|数字8| 5.7815e-12| 
|数字9| 2.5621e-09| 

为**数字4的概率是100%**，因此本次预测结果为4，正确。


4. 学习曲线

 训练集上cost 关于 迭代次数 的变化曲线

<img width="492" alt="" src="https://user-images.githubusercontent.com/19931702/110574313-db0c6700-8197-11eb-8280-4bbc50c84a1f.png">

 训练集和测试上的错误率关于 迭代epoths次数的变化曲线
   
<img width="489" alt="" src="https://user-images.githubusercontent.com/19931702/110574304-d5168600-8197-11eb-84b9-d5d951c7cad5.png">

训练集和测试上的准确率关于 迭代epoths次数的变化曲线

<img width="490" alt="" src="https://user-images.githubusercontent.com/19931702/110574297-d21b9580-8197-11eb-9dc2-939399adb03d.png">

训练集和测试上的cost关于 迭代epoths次数的变化曲线

<img width="491" alt="" src="https://user-images.githubusercontent.com/19931702/110574290-ce880e80-8197-11eb-8022-67940cd3962b.png">

训练每个epoth所花费的时间曲线变化

<img width="507" alt="" src="https://user-images.githubusercontent.com/19931702/110574274-c9c35a80-8197-11eb-8783-3a545fc0c846.png">



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
 
 

### 实现语言
octave6.1.0

    
    


#### 主要参考资料

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) 

    
