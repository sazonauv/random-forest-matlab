% Author: Viachaslau (Slava) Sazonau
% Week 2, Level 3
% Finding optimal gamma and cost parameters of
% SVM with RBF kernel via cross validation;
% Plotting confidence intervals


load heart;

n = 40; % gamma steps
m = 40; % cost steps
gRange = [0.01 0.1];
cRange = [2.5 3.5];
gStep = (gRange(2) - gRange(1))/n;
cStep = (cRange(2) - cRange(1))/m;
    
folds = 5;
foldSize = length(data)/folds;
repeat = 10;

dataR = data;
labelsR = labels;
gBestR = 0;
cBestR = 0;

errM = zeros(n,m,repeat);
errBestR = 1;
for r=1:repeat
    % shuffle the data
    len = length(labels);
    newInd = randperm(len);
    for i=1:len
        labelsR(i) = labels(newInd(i));
        dataR(i,:) = data(newInd(i),:);
    end
          
    cvErrors = zeros(folds-1);
%     minErr = 1;
        
    for i=1:n
        g = gRange(1) + i*gStep;        
        for j=1:m
            c = cRange(1) + j*cStep;            
            gStr = num2str(g);
            cStr = num2str(c);
            comArr = ['-t 2 -g ' gStr ' -c ' cStr];
            com = strcat(comArr);
            model = svm(com);                 %% builds an RBF svm with gamma and cost
            %cross-validation
            for v=0:folds-2
                %training
                dataK = dataR( [1 : v*foldSize, (v+1)*foldSize+1 : (folds-1)*foldSize] , : );
                labelsK = labelsR( [1 : v*foldSize, (v+1)*foldSize+1 : (folds-1)*foldSize] , : );
                model = model.train(dataK, labelsK);
                %validating
                dataV = dataR( v*foldSize+1 : (v+1)*foldSize , : );
                labelsV = labelsR( v*foldSize+1 : (v+1)*foldSize , : );
                cvErrors(v+1) = model.test(dataV, labelsV).err();
            end
            
            % find the model that performs best on average over the folds
            averErr = sum( cvErrors(:) )/(folds-1);

%             if minErr > averErr
%                 minErr = averErr;
%                 mini = i;
%                 minj = j;
% %                 modelBest = model;
%                 gBest = g;
%                 cBest = c;
%             end
            
            errM(i,j,r) = averErr;
            
        end
        
    end
    
%     %testing best model on this iteration of r
%             dataT = dataR( (folds-1)*foldSize+1 : folds*foldSize , : );
%             labelsT = labelsR( (folds-1)*foldSize+1 : folds*foldSize , : );            
%             errBest = modelBest.test(dataT,labelsT).err();
         
end

averages = zeros(n,m);
deviations = zeros(n,m);
errors = zeros(repeat,1);
for i=1:n
    for j=1:m
        for r=1:repeat
            errors(r) = errM(i,j,r);
        end
        averages(i,j) = sum( errors )/repeat;
        deviations(i,j) = norm( errors - averages(i,j) )/repeat;
    end
end

z95 = 1.96;
deviations = deviations.*(z95/sqrt(repeat)); 
%errorbar(averages, deviations)

minError = 1;
for i=1:n
    for j=1:m
        if (minError > averages(i,j))
            minError = averages(i,j);
            imin = i;
            jmin = j;
        end
    end    
end

minError
g = gRange(1) + imin*gStep
c = cRange(1) + jmin*cStep
