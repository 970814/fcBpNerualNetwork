
clear all;
close all;

load betterWB.txt

%图像大小
feaCou= 28*28;
% 设置网络每层的神经元数量
nnInfo = [feaCou 8 10];
% 网络层数
L = length(nnInfo);

[testLabels, testImages] = readTestData(10000);
testM = length(testLabels)
testImage = testImages(:,5);
figure(2);
imshow(reshape(testImage,28,28)');
testLabel = testLabels(2)
h = predict(betterWB,testImage,nnInfo)

testX=testImages;
% 将testY转换成10*testM 的矩阵，每列为一个样本x期望输出
testY = zeros(10, testM);
for i=1:testM,
    testY(testLabels(i)+1,i) =1;
end;

%计算测试样本的损失值
[weights,biases] = unboxWeightsAndBiases(betterWB,nnInfo);
testCost = costOf(testX,testY,weights,biases,L)
%计算分类错误率
[classifyErrRate,accuracy]=calClassifyErrRate(betterWB,testX,testLabels,nnInfo)








