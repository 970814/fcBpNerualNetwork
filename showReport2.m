function [] = showReport2(filename)
    close all;
    load(filename);

    figure(1);
    hold on;
    plot([0:(length(costs)-1)],costs);
    title('Cost');
    xlabel('iterations');
    ylabel('costs');
    hold off;

    figure(2);
    hold on;
    plot([0:(length(errRates)-1)],errRates);
    plot([0:(length(testErrRates)-1)],testErrRates);
    title('ErrorRate');
    legend('train','test');
    xlabel('epoths');
    ylabel('errRate');
    hold off;

    figure(3);
    hold on;
    plot([0:(length(accuArr)-1)],accuArr);
    plot([0:(length(testAccuArr)-1)],testAccuArr);
    title('Accuracy');
    legend('train','test');
    xlabel('epoths');
    ylabel('accuracy');
    hold off;


    figure(4);
    hold on;
    plot([0:(length(USCosts)-1)],USCosts);
    plot([0:(length(testCosts)-1)],testCosts);
    title('Costs');
    legend('train','test');
    xlabel('epoths');
    ylabel('costs');
    hold off;


    figure(5);
    hold on;
    plot([0:(length(perEpothCostTms)-1)],perEpothCostTms);
    title('CostTime');
    xlabel('epoth(th)');
    ylabel('costs');
    hold off;


%    save(fileReport,"nnInfo","betterWB","costs","testCosts","errRates","testErrRates","accuArr","testAccuArr");

[betterUSCost betterUSCostIndex]=min(USCosts)
[betterTestCost betterTestCostIndex]=min(testCosts)
[betterAccuracy betterAccuracyIndex]=max(accuArr);

[bettertestAccuracy bettertestAccuracyIndex]=max(testAccuArr);

betterAccuracy
betterAccuracyEpoths=betterAccuracyIndex
nowTestAcc=testAccuArr(betterAccuracyIndex)

bettertestAccuracy
bettertestAccuracyEpoths=bettertestAccuracyIndex
nowTrainAcc = accuArr(bettertestAccuracyIndex)
end;


