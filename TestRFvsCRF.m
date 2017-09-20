load heart;

% Number of data shuffles
repeat = 10;
dataR = data;
labelsR = labels;

[m n] = size(data);
mhalf = floor(m/2);
dataTrain = zeros(mhalf,n);
labelsTrain = zeros(mhalf,1);

dataTest = zeros(mhalf,n);
labelsTest = zeros(mhalf,1);

trees = 7;
fearures = 4;

resultsRF = zeros(2*repeat,1);
resultsCRF = zeros(2*repeat, 1);

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
        dataTest(i,:) = dataR(mhalf+i,:);
        labelsTest(i) = labelsR(mhalf+i);
    end
    
    % build a Random Forest
    rf1 = ROCRandomForest(dataTrain, labelsTrain);
    rf1.maxTrees = trees;
    rf1.fracSize = fearures;
    rf1.train();
    rf1.test(dataTest, labelsTest);
    resultsRF( (r-1)*2 + 1 ) = rf1.err();
    rf1.err()
    
    % build a Random Forest
    rf2 = ROCRandomForest(dataTest, labelsTest);
    rf2.maxTrees = trees;
    rf2.fracSize = fearures;
    rf2.train();
    rf2.test(dataTrain, labelsTrain);
    resultsRF( (r-1)*2 + 2 ) = rf2.err();
    rf2.err()
    
    % build a Combinatorial Random Forest
    rf1.trainGroups();
    rf1.testByGroups(dataTest, labelsTest);
    resultsCRF( (r-1)*2 + 1 ) = rf1.err();
    rf1.err()
    
    % build a Combinatorial Random Forest    
    rf2.trainGroups();
    rf2.testByGroups(dataTrain, labelsTrain);
    resultsCRF( (r-1)*2 + 2 ) = rf2.err();
    rf2.err()
        
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
errors = [mean(resultsRF), mean(resultsCRF)];
intervals = [std(resultsRF)*(z95/sqrt(repeat)), std(resultsCRF)*(z95/sqrt(repeat))];

figure;
bar(errors, 0.1, 'g', 'grouped')
grid on
title('Average errors and confidence intervals (95%)')
xlabel('1 - Random Forest; 2 - Combinatorial Random Forest')
ylabel('Error')
hold all;
errorbar(errors, intervals, 'xr','LineWidth',2,...
    'MarkerFaceColor','b',...
    'MarkerSize',5)

