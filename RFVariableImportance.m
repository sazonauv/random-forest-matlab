load breast;

% number of trees
m = 15;
% number of features in fraction chosen randomly
n = 5;
% Data shuffles
repeat = 20;

dataR = data;
labelsR = labels;

[rows cols] = size(data);
rowshalf = floor(rows/2);
dataTrain = zeros(rowshalf,cols);
labelsTrain = zeros(rowshalf,1);
dataTest = zeros(rowshalf,cols);
labelsTest = zeros(rowshalf,1);

vi = zeros(2*repeat, cols);

for r=1:repeat
    
    % shuffle the data
    len = length(labels);
    newInd = randperm(len);
    for i=1:len
        labelsR(i) = labels(newInd(i));
        dataR(i,:) = data(newInd(i),:);
    end
    for k=1:rowshalf
        dataTrain(k,:) = dataR(k,:);
        labelsTrain(k) = labelsR(k);
        dataTest(k,:) = dataR(rowshalf+k,:);
        labelsTest(k) = labelsR(rowshalf+k);
    end
        
    % build a Random Forest
    rf1 = ROCRandomForest(dataTrain, labelsTrain);
    rf1.maxTrees = m;
    rf1.fracSize = n;
    rf1.train();
    vi((r-1)*2+1, :) = rf1.estimateVI(dataTest, labelsTest);
    
    % build a Random Forest
    rf2 = ROCRandomForest(dataTest, labelsTest);
    rf2.maxTrees = m;
    rf2.fracSize = n;
    rf2.train();   
    vi((r-1)*2+2, :) = rf2.estimateVI(dataTrain, labelsTrain);
    
end

avervi = sum(vi)/(2*repeat);
z95 = 1.96;
devvi = zeros(1, cols);
for i=1:cols
    devvi(i) = std(vi(:, i) - avervi(i))*z95/sqrt(2*repeat);
end

figure
errorbar(avervi, devvi, 'r*','LineWidth',2,...
                'MarkerEdgeColor','r',...
                'MarkerFaceColor','r',...
                'MarkerSize',5)
title('Variable Importance via RF')
xlabel('Variable Index')
ylabel('Variable Importance')
grid on
hold all

