function h = predict(betterWB,X,nnInfo)
%    将wb转换成可用weights 和biases
    [weights,biases] = unboxWeightsAndBiases(betterWB,nnInfo);
%    网络层数
    L = length(nnInfo);
%    向前传播算法计算预测值
    a = X;
    for l = 2:L,
        z = weights{l} * a + biases{l};
%        a = sigmoid(z);
        if l < L,
            a = sigmoid(z);
        else
            % 最后一层不应用sigmoid函数
%            进行softmax 映射转换成概率分布
%            z 是10*m的矩阵
            t = e.^(z);
%            归一化
            a = t ./ sum(t,1);
        end;
    end;
    h = a;
end;