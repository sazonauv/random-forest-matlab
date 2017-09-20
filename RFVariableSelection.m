load breast;
[rows cols] = size(data);

% number of trees
m = 7;
% max number of features
nmax = cols;
n = floor(sqrt(cols));

elims = 3;
elimPart = 0.50;
% Line patterns
lines = [':rv', ':mx', ':bo'];

% Data shuffles
repeat = 20;
curveNum = 2*repeat;
ROCs = cell(1, curveNum);

dataR = data;
labelsR = labels;

rowshalf = floor(rows/2);
dataTrain = zeros(rowshalf,cols);
labelsTrain = zeros(rowshalf,1);
dataTest = zeros(rowshalf,cols);
labelsTest = zeros(rowshalf,1);

selFeatures = zeros(1, cols);
for f=1:cols
    selFeatures(f) = f;
end

numBest = floor((1-elimPart)*cols);
numWorst = cols - numBest;
bestSelection = zeros(1, numBest);
worstSelection = zeros(1, numWorst);

figure
grid on
axis([0 1 0 1])
hold all

for el=1:elims
    % number of features in fraction chosen randomly
    
    
    vi = zeros(1, nmax);
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
        rf1.selFeatures = selFeatures;
        rf1.train();
        ROCs{(r-1)*2+1} = rf1.generateROC(dataTest, labelsTest);
        
        % build a Random Forest
        rf2 = ROCRandomForest(dataTest, labelsTest);
        rf2.maxTrees = m;
        rf2.fracSize = n;
        rf2.selFeatures = selFeatures;
        rf2.train();        
        ROCs{(r-1)*2+2} = rf2.generateROC(dataTrain, labelsTrain);
        
        vi = vi + (rf1.estimateVI(dataTest, labelsTest) + rf2.estimateVI(dataTrain, labelsTrain))/2;
        
    end
    
    
    
    % Plot ROC curve for a Random Forest
    [averROC, deviations] = ROCRandomTree.averageThresholdROC(ROCs, 15);
    
    hold all
    plot(averROC(:,1), averROC(:,2), lines(el), 'LineWidth',2*(el - 1) + 1,...
        'MarkerEdgeColor','r',...
        'MarkerFaceColor','r',...
        'MarkerSize',10)
    
    errorbar(averROC(:,1), averROC(:,2), deviations(:,1))
    
    herrorbar(averROC(:,1), averROC(:,2), deviations(:,2))
    hold off
    
    if (el == 1)
        % Eliminate less important features
        nmax = numBest;
        
        vi = vi/repeat;
        [~, inds] = sort(vi, 'descend');
        
        for ind=1:nmax
            bestSelection(ind) = selFeatures(inds(ind));
        end
        
        for ind=nmax+1:cols
            worstSelection(ind-nmax) = selFeatures(inds(ind));
        end
        
        selFeatures = bestSelection;
        
    else if (el == 2)
            selFeatures = worstSelection;
            nmax = numWorst;
        end
        
    end
    
end


