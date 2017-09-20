load breast;

% number of features in fraction chosen randomly
n = 30; %length(data(1,:));
step = 1; %ceil(log2(n));
% Number of data shuffles
repeat = 20;

dataR = data;
labelsR = labels;

[rows cols] = size(data);
halfrows = floor(rows/2);
dataTrain = zeros(halfrows,cols);
labelsTrain = zeros(halfrows,1);
dataTest = zeros(halfrows,cols);
labelsTest = zeros(halfrows,1);

for r=1:repeat
    r
    % shuffle the data
    len = length(labels);
    newInd = randperm(len);
    for i=1:len
        labelsR(i) = labels(newInd(i));
        dataR(i,:) = data(newInd(i),:);
    end
    for k=1:rows/2
        dataTrain(k,:) = dataR(k,:);
        labelsTrain(k) = labelsR(k);
        dataTest(k,:) = dataR(halfrows+k,:);
        labelsTest(k) = labelsR(halfrows+k);
    end
            
for i=1:n/step       
            %cross-validation
            tree1 = RandomTree(dataTrain, labelsTrain);
            tree1.fracSize = (i-1)*step+1;
            tree1.train();
            tree1.test(dataTest, labelsTest);
            err1 = tree1.err();
            
            %cross-validation
            tree2 = RandomTree(dataTest, labelsTest);
            tree2.fracSize = (i-1)*step+1;           
            tree2.train();            
            tree2.test(dataTrain, labelsTrain);
            err2 = tree2.err();
            
            i
            averErr = (err1 + err2)/2
            errM(i,r) = averErr;        
end
end

averages = zeros(1, n/step);
deviations = zeros(1, n/step);
errors = zeros(1, repeat);
for i=1:n/step;        
        errors(:) = errM(i, :);        
        averages(i) = mean(errors);
        deviations(i) = std(errors);
end

z95 = 1.96;
deviations = deviations*(z95/sqrt(repeat));

figure;
errorbar(averages, deviations, '-xr','LineWidth',2,...
                'MarkerEdgeColor','r',...
                'MarkerFaceColor','r',...
                'MarkerSize',5)
grid on
title('Cross validation for a single decision tree')
xlabel('Number of features in a fraction')
ylabel('Error')
hold all;


% find the best model
minError = 1;
minDev = 1;
for i=1:n/step;    
        if (minError*minDev > averages(i)*deviations(i))
            minError = averages(i);
            minDev = deviations(i);
            imin = i;            
        end  
end

minError
minDev
fBest = imin
