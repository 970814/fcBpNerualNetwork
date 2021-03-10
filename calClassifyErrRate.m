function [classifyErrRate,accuracy]=calClassifyErrRate(betterWB,testX,testLabels,nnInfo)
%计算分类错误率

h = predict(betterWB,testX,nnInfo);
testM = size(testX,2);
pres = [];
% 将h转换成数字型
for i =1:testM,
    [v,index]=max(h(:,i));
    pres = [pres;index-1];
end;

classifyErrRate = length(find(pres ~= testLabels))*1.0 / testM;

accuracy = 1-classifyErrRate;


