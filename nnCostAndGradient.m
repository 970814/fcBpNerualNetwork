function [cost,g] = nnCostAndGradient(X,y,wb,nnInfo)

%    wb为列向量
%    nnInfo 类似 nnInfo= [784 16 16 10];
%    unboxing wb 成 多个weights，biases

    [weights,biases] = unboxWeightsAndBiases(wb,nnInfo);
%   网络层数
    L = length(nnInfo);


%    使用随机梯度下降

    global stochasticGradient;
    if stochasticGradient,
        global trainSet;
        X= trainSet;
        global trainY;
        y=trainY;

    end;

%  反向传播算法计算C关于w和b的梯度
    [cost, gw, gb] = backPropagationAlgorithm(X,y,weights,biases,L);
%cost
%记录costs
global costs;
costs = [costs cost];


% 计算在测试集训练集上各项指标
global plotTest;
plotTest=false;
if plotTest
    global testX;
    global testY;
    % 计算在测试集上的损失值
    testCost = costOf(testX,testY,weights,biases,L);
    global testCosts;
    testCosts = [testCosts testCost];

    global errRates;
    global accuArr;
%    计算拟合度建立在全部训练样本上
    global labels;
    global images;
    %记录拟合度
    [errRate,accuracy]=calClassifyErrRate(wb,images,labels,nnInfo);
    errRates= [errRates errRate];
    accuArr= [accuArr accuracy];
     %记录测试集上的准确率、错误率
    global testErrRates;
    global testAccuArr;
    global testLabels;
    [testErrRate,testAccuracy]=calClassifyErrRate(wb,testX,testLabels,nnInfo);
    testErrRates= [testErrRates testErrRate];
    testAccuArr= [testAccuArr testAccuracy];

end;

%   梯度检测
    gradientCheck = false;

    if gradientCheck,
        err=false;
        epsilon = 0.0001;
        maxErrDiff = 0.0001
        wc = 0;
        bc = 0;
        for l = 2:L,


%           检测 weights梯度是否正确
            [R,C]= size(weights{l})
            disp('开始检测 weights')
            for r=1:R,
                for c=1:C,

                    weights{l}(r,c)=weights{l}(r,c) + epsilon;
                    costB = costOf(X,y,weights,biases,L);
                    weights{l}(r,c)=weights{l}(r,c) - 2 * epsilon;
                    costA = costOf(X,y,weights,biases,L);
%                    使用双侧差分求偏导数
                    pdw = (costB-costA)/(2*epsilon);
                    if abs(pdw - gw{l}(r,c)) > maxErrDiff,
                        disp(sprintf('偏导数检测错误在第%d层的第%d行第%d列, 期望 %f, 实际是 %f, 差距是 %f',l,r,c,pdw,gw{l}(r,c),abs(pdw - gw{l}(r,c))));
%              当检测到第一个错误时，将停止检测
                        err=true;
                        break;
                    else
                        wc = wc+1;
%                        disp(sprintf('偏导数检测正确在第%d层的第%d行第%d列, 期望 %f, 实际是 %f, 差距是 %f',l,r,c,pdw,gw{l}(r,c),abs(pdw - gw{l}(r,c))));
                    end;
%                    还原权重值
                    weights{l}(r,c)=weights{l}(r,c) +  epsilon;

                end;
%              当检测到第一个错误时，将停止检测
                if err,
                    break;
                end;
            end;
%            当检测到第一个错误时，将停止检测
            if err,
                break;
            end;
%            检测biases梯度是否正确
            N = length(biases{l})
            disp('开始检测 biases')
            for n=1:N,
                 biases{l}(n)=biases{l}(n) + epsilon;
                 costB = costOf(X,y,weights,biases,L);
                 biases{l}(n)=biases{l}(n) - 2 * epsilon;
                 costA = costOf(X,y,weights,biases,L);
                 %                    使用双侧差分求偏导数
                 pdb = (costB-costA)/(2*epsilon);
                 if abs(pdb - gb{l}(n)) > epsilon,
                     disp(sprintf('偏导数检测错误在第%d层的第%d个, 期望 %f, 实际是 %f, 差距是 %f',l,n,pdb,gb{l}(n),abs(pdb - gb{l}(n))));
%            当检测到第一个错误时，将停止检测
                     err=true;
                     break;
                 else
                     bc = bc+1;
%                     disp(sprintf('偏导数检测正确在第%d层的第%d个, 期望 %f, 实际是 %f, 差距是 %f',l,n,pdb,gb{l}(n),abs(pdb - gb{l}(n))));
                 end;
%                    还原biases值
                 biases{l}(n)=biases{l}(n) + epsilon;
            end;
%            当检测到第一个错误时，将停止检测
            if err,
                break;
            end;
        end;
        disp(sprintf('%d 个w的偏导数计算正确,%d 个b的偏导数计算正确',wc,bc));
        if err,
            disp('检测到梯度计算错误');
        else
%            gw
%            gb
            disp('检测到梯度计算完全正确');
        end;



    end;

%    将所有的梯度转换成一个列向量,给 fminunc等高级优化函数使用
    g = [];
    for l=2:L,
        g=[g; gw{l}(:); gb{l}(:)];
    end;

end;