
clear all;
close all;

maxCount = 50000;
global labels;
global trainLabels;
global images;
[labels, images] = readTrainData(maxCount);
images = images / 255;
trainLabels= labels;
% 样本数量
m=length(labels);
size(labels)
size(images)
rows = 28;
columns = 28;
feaCou = rows*columns;

%for i=1:m,
%
%   imshow(reshape(images(:,i),28,28)')
%   labels(i)
%   pause(0.5)
%
%end;

% 设置网络每层的神经元数量
%nnInfo = [feaCou 8 10];
%nnInfo = [feaCou 10 10];
%nnInfo = [feaCou 40 10];
%nnInfo = [feaCou 48 10];
nnInfo = [feaCou 200 10]
% 网络层数
L = length(nnInfo);

%计算网络参数数量
wbSize = 0;
for i=2:L,
    wbSize = wbSize + nnInfo(i)*nnInfo(i-1)+ nnInfo(i);
end;
%epsilon_init = 0.12;
%随机初始化参数值 ,且属于(-epsilon_init,epsilon_init)
%wb =rand(wbSize,1) * 2 * epsilon_init - epsilon_init;
%wb =rand(wbSize,1) ;
% 使用方差为 1/k 的高斯分布生成w，方差为1 的高斯分布生成b
[wb] = initailizeWeightsAndBiases(nnInfo);


% 将y转换成10*m 的矩阵，每列为一个样本x期望输出
y = zeros(10, m);
for i=1:length(labels),
    y(labels(i)+1,i) =1;
end;


X= images;

%    记录迭代过程中的cost值
global costs;
costs=[];


global plotTest;
plotTest=false;

% 载入测试集，并设置为全局变量


[testLabels, testImages] = readTestData(10000);
testImages = testImages / 255;
testM=length(testLabels);
global testX;
testX = testImages;
global testY;
testY = zeros(10, testM);
for i=1:testM,
    testY(testLabels(i)+1,i) =1;
end;



if plotTest,
    global testCosts;
    testCosts=[];
    global testLabels;



    global errRates;
    global accuArr;
    global testErrRates;
    global testAccuArr;
    errRates=[];
    accuArr=[];
    testErrRates=[];
    testAccuArr=[];
end;

%


costFunction = @(wb) nnCostAndGradient(X,y,wb,nnInfo);

%% 为fmincg高级优化函数设置参数
%options = optimset('MaxIter', 300);
%%  为fminunc函数设置参数
%%options = optimset('GradObj', 'on', 'MaxIter', 400);
%% 梯度下降的高级优化算法寻找局部最优解
%
start = time();
%[betterWB, cost, info] = ...
%	fmincg(@(t)costFunction(t), wb, options);


USCosts=[];
testCosts=[];
errRates=[];
accuArr=[];
testErrRates=[];
testAccuArr=[];
perEpothCostTms=[];

% 随机梯度下降法
global stochasticGradient;
stochasticGradient = true;
options = optimset('MaxIter', 1);
% 训练批次
epoths = 50;
% 迷你训练批次大小
miniBatchSize = 10;
betterWB = wb;
for e = 1:epoths,
    seg = randperm(m / miniBatchSize);
    t0=time();
    for i = 1:length(seg),
%         使用全局变量 的迷你训练批次 x
        global trainSet;
        global trainY;
        trainSet = X(:,((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
        trainY = y(:,((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
        trainLabels = labels(((seg(i)-1)*miniBatchSize+1) :(seg(i)*miniBatchSize));
        [betterWB, cost, info] = ...
        	fmincg(@(t)costFunction(t), betterWB, options);
    end;
      disp(sprintf('第%d个epoth结束',e));

     [weights,biases] = unboxWeightsAndBiases(betterWB,nnInfo);
 %    每个epoth 后，计算在整个训练样本上的cost
    USCost = costOf(X,y,weights,biases,L)
    USCosts=[USCosts USCost];
%    每个epoth 后，计算在整个测试样本上的cost
    testCost = costOf(testX,testY,weights,biases,L)
    testCosts=[testCosts testCost];


%    每个epoth 后，计算在整个训练样本上的准确率
    [errRate,accuracy]=calClassifyErrRate(betterWB,X,labels,nnInfo)
    errRates= [errRates errRate];
    accuArr= [accuArr accuracy];
%    每个epoth 后，计算在整个测试样本上的准确率
    [testErrRate,testAccuracy]=calClassifyErrRate(betterWB,testX,testLabels,nnInfo)
    testErrRates= [testErrRates testErrRate];
    testAccuArr= [testAccuArr testAccuracy];
%   记录每个epoth训练的花费时间
    perEpothCostTm=time()-t0
    perEpothCostTms = [perEpothCostTms   perEpothCostTm];
end;




trainCostTime = time() - start

%[betterUSCost betterUSCostIndex]=min(USCosts)
%[betterTestCost betterTestCostIndex]=min(testCosts)
%[betterAccuracy betterAccuracyIndex]=max(accuArr)
%[bettertestAccuracy bettertestAccuracyIndex]=max(testAccuArr)



%filename = sprintf('betterWB %s.txt',datestr(now))

%save(filename,"betterWB");

%plot([0:(length(costs)-1)],costs);
%hold on;
%plot([0:(length(testCosts)-1)],testCosts);
%info


fileReport = sprintf('report/report2/fileReport %s.txt',datestr(now))

save(fileReport,"nnInfo","costs","testCosts","errRates","testErrRates","accuArr","testAccuArr","trainCostTime","USCosts","perEpothCostTms","betterWB");


showReport2(fileReport);



% 不同超参数得到的训练结果报告
% 1. 未优化的bp神经网络
% 2. 优化算法改变 、将fminunc 改成fmincg
% 3. 参数随机初始化，使用方差为 1/k 的高斯分布生成w，方差为1 的高斯分布生成b
% 4. batchGD 改成 随机梯度下降
% 5. 数据归一化
% 6. [784 8 10] 修改为 [784 10 10]
% 7. [784 10 10] 修改为 [784 30 10], min-batch itertaion 1 增加到2
% 8. [784 30 10], min-batch itertaion 2 修改为1
% 9. 修改 cost 函数  为 分段计算 ;以消除0*inf=NaN问题。
% 10.  [784 30 10] 修改为[784 40 10]    训练集拟合度99.01%, 测试集准确率96.49%，在训练了39个epoth获得
% 11. 10得到结果图，train和test 在cost 和accuracy上的差距随着训练次数增大而增大，存在一定的过度拟合，
%使用L2正则化,lambda=0.01,[784 40 10] 修改为[784 48 10],   训练集拟合度93.42%, 测试集准确率93.79%，在训练了33个epoth获得
% 12. 11得到结果图,表明了具有较高的泛化能力，但存在一定的欠拟合， [784 48 10] 修改为[784 64 10]
%                                     训练集拟合度93.77%, 测试集准确率94.25%，在训练了26个epoth获得
% 13. lambda=0.001                    训练集拟合度97.19%, 测试集准确率96.79%，在训练了36个epoth获得
% 14. [784 64 10] 修改为[784 90 10]    训练集拟合度97.47%, 测试集准确率96.94%，在训练了23个epoth获得
% 15. [784 90 10] 修改为[784 100 10]   训练集拟合度97.43%, 测试集准确率97.13%，在训练了54个epoth获得
% 16. [784 100 10] 修改为[784 200 10]  训练集拟合度97.36%, 测试集准确率97.22%，在训练了47个epoth获得
% 17. lambda=0.00005                  训练集拟合度99.05%, 测试集准确率97.98%，在训练了18个epoth获得
% 18. lambda=0.000001                 训练集拟合度99.71%, 测试集准确率98.08%，在训练了23个epoth获得
% 19. [784 200 10] 修改为[784 256 10]  训练集拟合度99.66%, 测试集准确率98.23%，在训练了17个epoth获得
% 20. lambda=0.00001                  训练集拟合度99.48%, 测试集准确率98.16%，在训练了22个epoth获得
% 21. [784 256 10] 修改为[784 300 10]  训练集拟合度99.61%, 测试集准确率98.24%，在训练了40个epoth获得
% 22. lambda=0.0000001                训练集拟合度99.86%, 测试集准确率98.32%，在训练了28个epoth获得
% 23. [784 300 10] 修改为[784 800 10]  训练集拟合度99.70%, 测试集准确率98.38%，在训练了28个epoth获得
% 24. [784 800 10] 修改为[784 200 10] lambda=0.00000001  训练集拟合度99.93%, 测试集准确率98.24%，在训练了50个epoth获得
%   上面，分别对应report内按时间排序的文件，



