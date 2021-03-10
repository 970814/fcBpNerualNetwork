function [] = showReport(filename)
    close all;
    load(filename);

    figure(1);
    hold on;
    plot([0:(length(costs)-1)],costs);
    plot([0:(length(testCosts)-1)],testCosts);
    title('Cost');
    legend('train','test');
    xlabel('iterations');
    ylabel('costs');
    hold off;

    figure(2);
    hold on;
    plot([0:(length(errRates)-1)],errRates);
    plot([0:(length(testErrRates)-1)],testErrRates);
    title('ErrorRate');
    legend('train','test');
    xlabel('iterations');
    ylabel('errRate');
    hold off;

    figure(3);
    hold on;
    plot([0:(length(accuArr)-1)],accuArr);
    plot([0:(length(testAccuArr)-1)],testAccuArr);
    title('Accuracy');
    legend('train','test');
    xlabel('iterations');
    ylabel('accuracy');
    hold off;


    figure(4);
    hold on;
    plot([0:(length(USCosts)-1)],USCosts);
    title('USCosts');
    xlabel('epoths');
    ylabel('costs');
    hold off;


%    save(fileReport,"nnInfo","betterWB","costs","testCosts","errRates","testErrRates","accuArr","testAccuArr");


end;