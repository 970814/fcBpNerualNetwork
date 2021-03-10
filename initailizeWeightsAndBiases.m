function [wb] = initailizeWeightsAndBiases(nnInfo)
  %
%   该初始化算法确实取得了一个非常好效果，
%    对于[784 8 10] 的网络，
%    1.训练速度加快，从 860s加速到722.7s       ->  提升了 15.97%
%    2.cost降的更低，从 2.7587 降低到 1.6204  ->  提升了 41.26%
%    3.准确率相应提升,详细 查看 fileReport 28-Feb-2021 20:56:07.txt、fileReport 28-Feb-2021 20:27:09.txt

% 独立随机变量和的方差，是每个独立随机变量方差的和
    L = length(nnInfo);
    wb = [];
    for i = 2:L,
         k = nnInfo(i);
%         使用方差为 1/k 的高斯分布生成w，方差为1 的高斯分布生成b
         wb = [wb;1/sqrt(k) * randn(nnInfo(i-1)*nnInfo(i),1) ;randn(nnInfo(i),1)];
    end;

end;

