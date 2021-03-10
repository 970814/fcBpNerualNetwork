function [cost, gw, gb] = backPropagationAlgorithm(X,y,weights,biases,L)
%   X 输入数据矩阵，由m个列向量组成，每个列向量为一个输入样本
%   y 标签，期待输出
%   weights  权重，weights{l} 代表第l层的权重
%   biases  偏差项，biases{l} 代表第l层的偏差
%    L     神经网络的层数

%    向前传播算法
%    a{l}为第l层的输出激活值
%    z{l}为第l层的输入激活值
    a={X};
    z={};
    for l = 2:L,
        z{l} = weights{l} * a{l-1} + biases{l};

%        a{l} = sigmoid(z{l});

        if l < L,
            a{l} = sigmoid(z{l});
        else
            % 最后一层不应用sigmoid函数
%            进行softmax 映射转换成概率分布
%            z 是10*m的矩阵
            t = e.^(z{l});
%            归一化
            a{l} = t ./ sum(t,1);
        end;


    end;
%    样本数量
    m = size(X,2);
%    cost-entropy 计算 损失函数
%    cost = sum(sum(-y .* log(a{L}) - (1 - y) .* log(1 - a{L}))) / m ;

% 因为一般的cross-entropy 写法将导致 出现0*Inf  = NaN ，改进成分段计算
%    y2 = y(:);
%    a2 = a{L}(:);
%    eq0 = find(y2==0);
%    neq0 = find(y2~=0);
%    eq0Cost = - (1 - y2(eq0)) .* log(1 - a2(eq0));
%    neq0Cost = - y2(neq0) .* log(a2(neq0));
%    cost = (sum (eq0Cost) + sum(neq0Cost))/m;

% softmax 对应的损失函数对于单个样本 cost = -sum(y .* log(a)),为了加快运算，实际上这只是单个项并非累加项，
%因为只有y_k为1,cost = -y_k *log(a_k) = -log(a_k),
    y2 = y(:);
    a2 = a{L}(:);
    eq1 = find(y2==1);
    cost = sum(sum(-log(a2(eq1))))/m;



%    计算损失函数C关于第L层（最后一层）的输入激活值z的偏导数
    Delta={};
    Delta{L} = a{L} - y;
    for l = (L-1):-1:2,
%        反向传播误差
        Delta{l} = (weights{l+1}' * Delta{l+1}) .* sigmoidDerivatives(z{l});
    end;
    % L2正则化参数
    lambda = 0.00000001;
  %    计算关于每层w和b的偏导数
    for l = L:-1:2,
        gw{l} =  Delta{l} * a{l-1}' ./ m;

        % L2正则化
        gw{l} = gw{l} + lambda/m * weights{l};

        gb{l} = sum(Delta{l},2) ./m;
    end;


end;





