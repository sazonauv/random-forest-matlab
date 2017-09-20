load heart;

% Number of data shuffles
repeat = 25;
dataR = data;
labelsR = labels;

[m n] = size(data);
dataTrain = zeros(m/2,n);
labelsTrain = zeros(m/2,1);

dataTest = zeros(m/2,n);
labelsTest = zeros(m/2,1);

trees = 15;
fearures = 5;

g = 0.0325;
c = 3.3250;
gStr = num2str(g);
cStr = num2str(c);
comArr = ['-t 2 -g ', gStr, ' -c ', cStr];
com = strcat(comArr);

resultsRF = zeros(2*repeat,1);
resultsSVM = zeros(2*repeat, 1);

for r=1:repeat
    r
    % shuffle the data
    newInd = randperm(m);
    for i=1:m
        labelsR(i) = labels(newInd(i));
        dataR(i,:) = data(newInd(i),:);
    end
    
    for i=1:m/2
        dataTrain(i,:) = dataR(i,:);
        labelsTrain(i) = labelsR(i);
        dataTest(i,:) = dataR(m/2+i,:);
        labelsTest(i) = labelsR(m/2+i);
    end
    
    % build a Random Forest
    rf = RandomForest(dataTrain, labelsTrain);
    rf.maxTrees = trees;
    rf.fracSize = fearures;
    rf.train();
    rf.test(dataTest, labelsTest);
    resultsRF( (r-1)*2 + 1 ) = rf.err();
    
    % build a Random Forest
    rf = RandomForest(dataTest, labelsTest);
    rf.maxTrees = trees;
    rf.fracSize = fearures;
    rf.train();
    rf.test(dataTrain, labelsTrain);
    resultsRF( (r-1)*2 + 2 ) = rf.err();
    
    % builds an RBF svm with gamma and cost
    rbf = svm(com);
    rbf = rbf.train(dataTrain, labelsTrain);
    
    resultsSVM( (r-1)*2 + 1 ) = rbf.test(dataTest, labelsTest).err();
    
    % builds an RBF svm with gamma and cost
    rbf = rbf.train(dataTest, labelsTest);
    
    resultsSVM( (r-1)*2 + 2 ) = rbf.test(dataTrain, labelsTrain).err();
    
end

% figure;
% plot(resultsRF,'--gs','LineWidth',2,...
%     'MarkerFaceColor','g',...
%     'MarkerSize',5)
% grid on
% title('Test errors')
% xlabel('Data shuffle number')
% ylabel('Error')
% hold all;
% plot(resultsSVM, '--bs','LineWidth',2,...
%     'MarkerFaceColor','b',...
%     'MarkerSize',5)
% hold off;

z95 = 1.96;
errors = [mean(resultsRF), mean(resultsSVM)];
intervals = [std(resultsRF)*(z95/sqrt(repeat)), std(resultsSVM)*(z95/sqrt(repeat))];

figure;
bar(errors, 0.1, 'g', 'grouped')
grid on
title('Average errors and confidence intervals (95%)')
xlabel('1 - Random Forest; 2 - SVM')
ylabel('Error')
hold all;
errorbar(errors, intervals, 'xr','LineWidth',2,...
    'MarkerFaceColor','b',...
    'MarkerSize',5)

