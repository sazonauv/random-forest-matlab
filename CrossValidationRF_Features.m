load breast;

% number of trees
m = 7;
% number of features in fraction chosen randomly
n = 30; %length(data(1,:));
step = 3; %ceil(log2(n));
% Number of data shuffles
repeat = 5;

dataR = data;
labelsR = labels;

[rows cols] = size(data);
halfrows = floor(rows/2);
dataTrain = zeros(halfrows,cols);
labelsTrain = zeros(halfrows,1);
dataTest = zeros(halfrows,cols);
labelsTest = zeros(halfrows,1);

errM = zeros(n/step, 1, repeat);

for r=1:repeat
    
    % shuffle the data
    len = length(labels);
    newInd = randperm(len);
    for i=1:len
        labelsR(i) = labels(newInd(i));
        dataR(i,:) = data(newInd(i),:);
    end
    for k=1:halfrows
        dataTrain(k,:) = dataR(k,:);
        labelsTrain(k) = labelsR(k);
        dataTest(k,:) = dataR(halfrows+k,:);
        labelsTest(k) = labelsR(halfrows+k);
    end
            
for i=1:n/step
        for j=1:1
            % build a Random Forest
            rf1 = RandomForest(dataTrain, labelsTrain);
            rf1.maxTrees = m;
            rf1.fracSize = i*step;    
            rf1.train();
            rf1.test(dataTest, labelsTest);
            err1 = rf1.err();
    
            % build a Random Forest
            rf2 = RandomForest(dataTest, labelsTest);
            rf2.maxTrees = m;
            rf2.fracSize = i*step;            
            rf2.train();
            rf2.test(dataTrain, labelsTrain);
            err2 = rf2.err();
            
            
            averErr = (err1 + err2)/2;
            errM(i,j,r) = averErr;
        end
end
end

averages = zeros(n/step, 1);
deviations = zeros(n/step, 1);
errors = zeros(repeat,1);
for i=1:n/step;
    for j=1:1
        for r=1:repeat
            errors(r) = errM(i,j,r);
        end
        averages(i,j) = sum( errors )/repeat;
        deviations(i,j) = norm( errors - averages(i,j) )/repeat;
    end
end

z95 = 1.96;
deviations = deviations.*(z95/sqrt(2*repeat));

figure;
errorbar(averages, deviations, '-xr','LineWidth',2)
grid on
title('Cross validation for Random Forest')
xlabel('Number of features in a fraction')
ylabel('Error')
hold all;


% find the best model
minError = 1;
minDev = 1;
for i=1:n/step;
    for j=1:1
        if (minError*minDev > averages(i,j)*deviations(i,j))
            minError = averages(i,j);
            minDev = deviations(i,j);
            imin = i;
            jmin = j;
        end
    end
end


