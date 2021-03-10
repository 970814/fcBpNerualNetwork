function [weights,biases] = unboxWeightsAndBiases(wb,nnInfo)
%    wb为列向量
%    nnInfo 类似 nnInfo= [784 16 16 10];
%    unboxing wb 成 多个weights，biases
        L = length(nnInfo);
        offset = 0;
        for l = 2:L,
            weights{l} = reshape(wb(offset + 1:offset + nnInfo(l) * nnInfo(l-1)), nnInfo(l) , nnInfo(l-1));
            offset = offset + nnInfo(l) * nnInfo(l-1);
            biases{l} = wb(offset + 1:offset + nnInfo(l) );
            offset = offset + nnInfo(l) ;
        end;
end;
