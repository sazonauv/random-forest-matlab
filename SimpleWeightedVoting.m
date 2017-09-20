load heart;
[rows cols] = size(data);

% number of trees
m = 15;
% number of features in fraction chosen randomly
n = 3
% max number of features
nmax = cols;
% Data shuffles
repeat = 10;

curveNum = 2*repeat;
ROCs1 = cell(1, curveNum);
ROCs2 = cell(1, curveNum);
ROCs3 = cell(1, curveNum);

dataR = data;
labelsR = labels;


rowshalf = floor(rows/2);
dataTrain = zeros(rowshalf,cols);
labelsTrain = zeros(rowshalf,1);
dataTest = zeros(rowshalf,cols);
labelsTest = zeros(rowshalf,1);

for r=1:repeat
    r
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
    rf1 = RandomForest(dataTrain, labelsTrain);
    rf1.maxTrees = m;
    rf1.fracSize = n;
    rf1.train();
    ROCs1{(r-1)*2+1} = rf1.generateROC(dataTest, labelsTest);
    
     % build a Random Forest
    rf2 = RandomForest(dataTest, labelsTest);
    rf2.maxTrees = m;
    rf2.fracSize = n;
    rf2.train();
    ROCs1{(r-1)*2+2} = rf2.generateROC(dataTrain, labelsTrain);
    
    % build a Random Forest
    rf1 = ROCRandomForest(dataTrain, labelsTrain);
    rf1.maxTrees = m;
    rf1.fracSize = n;
    rf1.train();    
    ROCs2{(r-1)*2+1} = rf1.generateROC(dataTest, labelsTest);
    
    % build a Random Forest
    rf2 = ROCRandomForest(dataTest, labelsTest);
    rf2.maxTrees = m;
    rf2.fracSize = n;
    rf2.train();    
    ROCs2{(r-1)*2+2} = rf2.generateROC(dataTrain, labelsTrain);
    
end

% Plot ROC curve for a Random Forest
[averROC1, deviations1] = ROCRandomTree.averageThresholdROC(ROCs1, 15);

figure
plot(averROC1(:,1), averROC1(:,2), '--b*','LineWidth',2,...
                'MarkerEdgeColor','b',...
                'MarkerFaceColor','b',...
                'MarkerSize',5)
grid on
axis([0 1 0 1])
hold all

errorbar(averROC1(:,1), averROC1(:,2), deviations1(:,1))
hold all

herrorbar(averROC1(:,1), averROC1(:,2), deviations1(:,2))
hold all

% Plot ROC curve for a Random Forest
[averROC2, deviations2] = ROCRandomTree.averageThresholdROC(ROCs2, 15);


plot(averROC2(:,1), averROC2(:,2), '--r*','LineWidth',2,...
                'MarkerEdgeColor','r',...
                'MarkerFaceColor','r',...
                'MarkerSize',5)
grid on
axis([0 1 0 1])
hold all

errorbar(averROC2(:,1), averROC2(:,2), deviations2(:,1))
hold all

herrorbar(averROC2(:,1), averROC2(:,2), deviations2(:,2))

