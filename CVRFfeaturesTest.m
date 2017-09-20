load heart;

% number of trees
m = 9;
% number of features in fraction chosen randomly
n = 13; %length(data(1,:));
step = 1; %ceil(log2(n));
% Number of data shuffles
repeat = 26;

dataR = data;
labelsR = labels;

[rows cols] = size(data);
dataTrain = zeros(rows/2,cols);
labelsTrain = zeros(rows/2,1);
dataTest = zeros(rows/2,cols);
labelsTest = zeros(rows/2,1);



averages = zeros(n/step, 1);
deviations = zeros(n/step, 1);
errors = zeros(repeat,1);
for i=1:n/step;

        for r=1:repeat
            errors(r) = errM(r, i);
        end
        averages(i) = sum( errors )/repeat;
        deviations(i) = norm( errors - averages(i) )/repeat;

end

z95 = 1.96;
deviations = deviations.*(z95/sqrt(repeat));

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

        if (minError*minDev > averages(i)*deviations(i))
            minError = averages(i);
            minDev = deviations(i);
            imin = i;
            
        end

end

minError
minDev
nBest = imin

