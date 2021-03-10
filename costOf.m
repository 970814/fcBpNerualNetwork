function [cost] = costOf(X,y,weights,biases,L)
%   X 输入数据矩阵，由m个列向量组成，每个列向量为一个输入样本
%   y 标签，期待输出
%   weights  权重，weights{l} 代表第l层的权重
%    biases  偏差项，biases{l} 代表第l层的偏差
%    L     神经网络的层数

%    向前传播算法
    a = X;
    for l = 2:L,
        z = weights{l} * a + biases{l};

        a = sigmoid(z);
    end;
%    样本数量
    m = size(a,2);
%    cross-entropy 计算 损失函数
%    cost = sum(sum(-y .* log(a) - (1 - y) .* log(1 - a))) / m ;

% 因为一般的cross-entropy 写法将导致 出现0*Inf  = NaN ，改进成分段计算
    y2 = y(:);
    a2 = a(:);
    eq0 = find(y2==0);
    neq0 = find(y2~=0);
    eq0Cost = - (1 - y2(eq0)) .* log(1 - a2(eq0));
    neq0Cost = - y2(neq0) .* log(a2(neq0));
    cost = (sum (eq0Cost) + sum(neq0Cost))/m;
% L2正则化
    lambda = 0.00000001;
    regularizationTerm = 0;
    for l = 2:L,
        regularizationTerm = regularizationTerm + sum(sum(weights{l} .^ 2));
    end;
    cost = cost + lambda /2/ m * regularizationTerm;

%    if isnan(cost),
%        save(yaNaN,"y","a");
%    end;
end;

%使用 cost = sum(sum(-y .* log(a) - (1 - y) .* log(1 - a))) / m ;该形式将
%出现NaN情况，
%如 a=[0 0.4 0.3 0.6 0.5]'
%   y=[0 1   0   1   1]'
%   改进办法是分段计算;
%

