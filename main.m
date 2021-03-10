

%载入模型
%load('report/report2/fileReport 10-Mar-2021 11:14:46.txt')


% 读取测试集数据
[testLabels, testImages] = readTestData(10000);
% x归一化
testImages = testImages / 255;
m = length(testImages);
for i=1:m,
   img=testImages(:,i);
   imshow(reshape(img,28,28)');
   expect = testLabels(i)
   %进行预测
   h = predict(betterWB,img,nnInfo)
   [probability,index]=max(h);
   probability
   predictValue = index-1
%   sumh= sum(h)
   pause;
end;




