function h = predict(betterWB,X,nnInfo)
%    将wb转换成可用weights 和biases
    [weights,biases] = unboxWeightsAndBiases(betterWB,nnInfo);
%    网络层数
    L = length(nnInfo);
%    向前传播算法计算预测值
    a = X;
    for l = 2:L,
        z = weights{l} * a + biases{l};
        a = sigmoid(z);
    end;
    h = a;
end;