classdef BROCRandomForest < handle
    %% RANDOMFOREST A class that implements a "random forest" machine learning algorithm.
    %
    % Author: Viachaslau (Slava) Sazonau
    % Project: Implementation and evaluation of Random Forest
    % COMP61011: Machine Learning and Data Mining
    % Date: 12-Octrober-2012
    %
    % Builds and trains a random forest for the given data set.
    % Uses class ROCRandomTree.
    
    % Properties
    properties        
        % Max number of trees allowed in the forest
        maxTrees;
        % Number of features in each fraction while splitting
        fracSize;
        % Randomized desicion trees (references):
        % First grown tree in the forest
        first;
        % Last grown tree in the forest
        last;
        % Current number of trees in the forest
        amount;
        % Testing error
        testError;
        % Data
        data;
        % Labels
        labels;
        % Target class for ROC analysis (i.e. positive)
        targetLabel;
        % Groups of trees
        groups;
        % Number of groups to be selected
        groupsNum;
        % Group weights
        gweights;
        % Class labels in the train set
        classes;
        % Group size
        groupSize;
    end
    
     % Constants
    properties (Constant)
        % Default number of trees in the forest
        defMaxTrees = 7;
        % Default target class label
        defTargetLabel = 1;
        % Default group size
        defGroupSize = 3;
        % Default number of groups to be selected
        defGroupsNum = 10;
        % Error messages
        error_max_size = 'Max size of the forest is exceeded, therefore adding is not possible.';
        error_empty = 'The forest contains no trees. Classification is not possible. Train the forest first.';
        error_outofbounds = 'Tree ID is out of forest bounds. ID must be lower or equal to forest.amount.';
        error_nogroups = 'No groups are created. Create the groups first.';
    end
    
    % Public methods
    methods
        
        % Constructor
        function forest = ROCRandomForest(data, labels)
                forest.maxTrees = ROCRandomForest.defMaxTrees;
                forest.fracSize = ROCRandomTree.defFracSize;
                forest.targetLabel = ROCRandomForest.defTargetLabel;
                forest.groupSize = ROCRandomForest.defGroupSize;
                forest.groupsNum = ROCRandomForest.defGroupsNum;
                forest.amount = 0;
                forest.data = data;
                forest.labels = labels;
                forest.classes = unique(forest.labels);
        end              
        
% Get/set section 
         % Get max number of trees allowed in the forest
         function maxTrees = get.maxTrees(forest)
             maxTrees = forest.maxTrees;
         end
         
         % Set max number of trees allowed in the forest
         function set.maxTrees(forest, maxTrees)
             forest.maxTrees = maxTrees;
         end
           
         % Set the number of features in each fraction while splitting
         function set.fracSize(forest, fracSize)
             forest.fracSize = fracSize;             
         end
         
         % Get the number of features in each fraction while splitting
         function fracSize = get.fracSize(forest)
             fracSize = forest.fracSize;             
         end
         
         % Get current amount of the forest (the number of trees in it)
         function amount = get.amount(forest)
             amount = forest.amount;
         end
            
         % Get the first tree grown in the forest
         function first = get.first(forest)
             first = forest.first;
         end
                  
         % Get the last tree grown in the forest
         function last = get.last(forest)
             last = forest.last;
         end
         
         % Get data
        function data = get.data(forest)
            data = forest.data;
        end
        
        % Set data
        function set.data(forest, data)
            forest.data = data;
        end
        
        % Get labels
        function labels = get.labels(forest)
            labels = forest.labels;
        end
        
        % Set labels
        function  set.labels(forest, labels)
            forest.labels = labels;
        end
         
        % Get target class label
        function targetLabel = get.targetLabel(forest)
            targetLabel = forest.targetLabel;
        end
        
        % Set target class label
        function  set.targetLabel(forest, targetLabel)
            forest.targetLabel = targetLabel;           
        end
        
% End of get/set section

        % Get a tree by ID
        function tree = getTree(forest, id)
            if (forest.amount <= 0)
                error(ROCRandomForest.error_empty);
            else if (id > forest.amount)
                    error(ROCRandomForest.error_outofbounds);
                else
                    i = 1;
                    tree = forest.first;
                    while (i < id)
                        tree = tree.next;
                        i = i + 1;
                    end
                end
            end
        end

        % Add a new grown tree to the forest
        function addTree(forest, tree)
            if (isempty(forest.last))                
                forest.first = tree;
                forest.last = tree;
                forest.amount = forest.amount + 1;
            else
                if (forest.amount < forest.maxTrees)                    
                    forest.last.next = tree;
                    tree.previous = forest.last;
                    forest.last = tree;
                    forest.amount = forest.amount + 1;
                else
                    error(ROCRandomForest.error_max_size);
                end
            end
        end
        
        % Grow and train the forest
        function train(forest)            
            for i=1:forest.maxTrees
                indexes = forest.bootstrap();
                tree = ROCRandomTree(forest.data, forest.labels);
                tree.fracSize = forest.fracSize;
                tree.targetLabel = forest.targetLabel;
                tree.trainIndex(indexes);
                tree.test(forest.data, forest.labels);            
                forest.addTree(tree);
            end            
        end
        
        % Train groups
        function trainGroups(forest)            
            forest.createGroups();            
            % Estimate weights
            forest.estimateWeights();
            % Select best groups
            forest.selectGroups();
        end
        
        % Classify a test example x using the forest
        function label = classify(forest, x)
            if (isempty(forest.first))
                error(ROCRandomForest.error_empty);
            else
                % Get class probabilities
                probs = zeros(size(forest.classes));
                for i=1:length(probs)
                    probs(i) = forest.getProbability(x, forest.classes(i));                    
                end                
                [~, ind] = max(probs);
                label = forest.classes(ind);                
            end            
        end
        
        % Classify a test example x using the forest
        function label = classifyByGroups(forest, x)
            if (isempty(forest.first))
                error(ROCRandomForest.error_empty);
            else
                % Get class probabilities
                probs = zeros(size(forest.classes));
                for i=1:length(probs)
                    probs(i) = forest.getProbabilityByGroups(x, forest.classes(i));                    
                end                
                [~, ind] = max(probs);
                label = forest.classes(ind);                
            end            
        end
        
        % Classify a test example x using the forest
        function label = classifyv00(forest, x)
            if (isempty(forest.first))
                error(ROCRandomForest.error_empty);
            else
                % Collect votes
                votes = zeros(forest.amount, 1);
                tree = forest.first;
                for i=1:forest.amount
                    votes(i) = tree.classify(x);
                    tree = tree.next;                    
                end
                % Take a major vote as a class label
                votesHist = ROCRandomForest.classHist(votes);
                [~, indVec] = max(votesHist);
                label = indVec(1) - 1;
            end            
        end
        
        % Test the forest classification performance
        function test(forest, dataT, labelsT)
            len = length(labelsT);
            if (len == 0)
                error(ROCRandomTree.error_no_input);                
            end                        
            err = 0;
            for i=1:len
                x = dataT(i,:);
                label = forest.classify(x);
                if (label ~= labelsT(i))
                    err = err + 1;
                end
            end
            % Get testing error
            err = err / len;
            forest.testError = err;
        end
        
         % Test the forest classification performance
        function testByGroups(forest, dataT, labelsT)
            len = length(labelsT);
            if (len == 0)
                error(ROCRandomTree.error_no_input);                
            end                        
            err = 0;
            for i=1:len
                x = dataT(i,:);
                label = forest.classifyByGroups(x);
                if (label ~= labelsT(i))
                    err = err + 1;
                end
            end
            % Get testing error
            err = err / len;
            forest.testError = err;
        end
        
        % Get testing error
        function testError = err(forest)
            testError = forest.testError;
        end
        
        % Get a bootstrap from the given data (performed with replacement)
        function [indexes] = bootstrap(forest)            
            len = length(forest.labels);
            indexes = randi(len, len, 1);                        
        end
        
        % Get class probability for example x
        function p = getProbability(forest, x, classLabel)
            tree = forest.first;
            weights = zeros(1, forest.amount);
            % Normalise weights
            i = 1;
            while (~isempty(tree))
                weights(i) = 1 - tree.err();
                tree = tree.next;
                i = i + 1;
            end
            weights = weights/sum(weights);
            % Estimate probability
            tree = forest.first;
            p = 0;
            i = 1;
            while (~isempty(tree))
                p = p + weights(i)*tree.getProbability(x, classLabel);
                tree = tree.next;
                i = i + 1;
            end            
        end
        
        % Get class probability for example x using groups
        function p = getProbabilityByGroups(forest, x, classLabel)           
            % Estimate probability            
            p = 0;
            for i=1:length(forest.groups(:, 1))                
                p = p + forest.gweights(i)*forest.getGroupProbability(x, classLabel, i);                
            end            
        end
        
        % Get class probability by the group
        function p = getGroupProbability(forest, x, classLabel, gnum)
            errsum = 0;            
            for i=1:forest.groupSize
                tree = forest.getTree(forest.groups(gnum, i));
                errsum = errsum + tree.err();
            end
            p = 0;           
            for i=1:forest.groupSize
                tree = forest.getTree(forest.groups(gnum, i));
                p = p + tree.getProbability(x, classLabel)*(1 - tree.err())/(forest.groupSize - errsum);                
            end            
        end
        
        % Update tree target labels
        function updateTargets(forest, target)
            if (forest.amount > 0)
                tree = forest.first;
                while (~isempty(tree))
                    tree.targetLabel = target;
                    tree = tree.next;                    
                end                
            end   
        end
        
        % Generate ROC curve for the forest
        function curve = generateROC(forest, dataT, labelsT)
            % Find the number of positives and negatives
            positives = 0;
            len = length(labelsT);
            for i=1:len
                if (labelsT(i) == forest.targetLabel)
                    positives = positives + 1;
                end
            end
            negatives = len - positives;
            % Build a ROC curve
            [scores ids] = forest.getScores(dataT, labelsT);            
            FP = 0;
            TP = 0;
            curve = zeros(len+1, 3);
            fprev = -1;
            i = 1;
            ind = 1;
            while (i <= len)
                if (scores(i) ~= fprev)
                    curve(ind, :) = [FP/negatives, TP/positives, scores(i)];
                    ind = ind + 1;
                    fprev = scores(i);
                end
                if (labelsT(ids(i)) == forest.targetLabel)
                    TP = TP + 1;
                else
                    FP = FP + 1;
                end
                i = i + 1;
            end
            curve(ind, :) = [FP/negatives, TP/positives, 0];
            curve = curve(1:ind, :);
        end
        
        % Generate scores for training data
        function [scores ids] = getScores(forest, dataT, labelsT)
            len = length(labelsT);
            scores = zeros(1, len);
            for i=1:len
                scores(i) = forest.getProbability(dataT(i, :), forest.targetLabel);                
            end
            [scores ids] = sort(scores, 'descend');
        end
        
        %GROUPS
        
        % Generate all k-combinations of a given set n - C(k, n)
        % where n - max number for the set of integers, i.e. 1..n
        function createGroups(forest)
            n = forest.amount;
            k = forest.groupSize;
            % Total quantity of possible combinations
            quan = factorial(n)/(factorial(k)*factorial(n-k));
            % Initialise groups array
            forest.groups = zeros(quan, k);            
            % Initialise a first combination
            comb = zeros(1, k);            
            for i=1:k
                comb(i) = i;
            end
            % Generate all k-combinations
            forest.groups(1, :) = comb;            
            q = 2;
            while (comb(1) < n-k+1)
                i = 2;
                while (i <= k && comb(i) < comb(i-1) + 2)
                    i = i + 1;
                end
                comb(i-1) = comb(i-1) + 1;
                for j=1:i-2
                    comb(j) = j;
                end
                forest.groups(q, :) = comb;
                q = q + 1;
            end
        end
        
        %Generate random groups
        function randomGroups(forest)
            gsize = forest.groupSize;
            tnum = forest.maxTrees;
            fnum = floor(tnum/gsize);
            gnum = forest.groupsNum;
            
            forest.groups = zeros(fnum, gsize);            
            divmax = -realmax();
            for i=1:gnum
                set = randperm(tnum);
                div = forest.groupAllDiversity(set);
                if (divmax < div)
                    divmax = div;
                    setmax = set;
                end
            end            
            for i=1:fnum
                forest.groups(i, :) = setmax((i-1)*gsize+1 : i*gsize);
            end
        end
        
        % Estimate group weights
        function estimateWeights(forest)            
            forest.gweights = zeros(1, length(forest.groups(:,1)));
            for i=1:length(forest.gweights)
                forest.gweights(i) = forest.groupDiversity(i);              
            end
            forest.gweights = forest.gweights/sum(forest.gweights);            
        end
        
        % Selects best groups
        function selectGroups(forest)
            [ws inds] = sort(forest.gweights, 'descend');            
            k = forest.groupSize;
            quan = forest.groupsNum;
            gs = zeros(quan, k);
            weights = zeros(quan, 1);
            for i=1:quan
                gs(i, :) = forest.groups(inds(i), :);
                weights(i) = ws(i);
            end
            forest.groups = gs;
            forest.gweights = weights/sum(weights);
        end
        
        % Estimate diversity between the trees in the group
        function div = groupDiversity(forest, gnum)           
            k = forest.groupSize;
            averInfHist = zeros(1, length(forest.data(1, :)));
            errsum = 0;
            for i=1:k
                tree = forest.getTree(forest.groups(gnum, i));
                averInfHist = averInfHist + tree.infHist;
                errsum = errsum + tree.err();
            end
            averInfHist = averInfHist/k;
            div = 0;
            for i=1:k
                tree = forest.getTree(forest.groups(gnum, i));
                div = div + pdist([tree.infHist; averInfHist], 'spearman')*(1 - tree.err())/(k - errsum);                    
            end
            div = 1 - errsum/k;%div/k;
        end
        
        % Estimate diversity between the trees in the group
        function div = groupAverageDiversity(forest, set, start)           
            k = forest.groupSize;
            averInfHist = zeros(1, length(forest.data(1, :)));
            errsum = 0;
            for i=start:start+k-1
                tree = forest.getTree(set(i));
                averInfHist = averInfHist + tree.infHist;
                errsum = errsum + tree.err();
            end
            averInfHist = averInfHist/k;
            div = 0;
            for i=start:start+k-1
                tree = forest.getTree(set(i));
                div = div + pdist([tree.infHist; averInfHist], 'spearman')*(1 - tree.err())/(k - errsum);                    
            end
            div = exp(div/k);
        end
        
        % Average diversity measure for all groups in the set
        function div = groupAllDiversity(forest, set)
            div = 0;
            for j=1:floor(forest.amount/forest.groupSize)
                div = div + groupAverageDiversity(forest, set, (j-1)*forest.groupSize+1);                
            end
            div = div/j;            
        end
        
        % Get error for group
        function err = getGroupError(forest, gnum, dataT, labelsT)            
            len = length(labelsT);
            if (len == 0)
                error(ROCRandomTree.error_no_input);                
            end                        
            err = 0;
            for i=1:len
                x = dataT(i,:);
                label = forest.getGroupLabel(gnum, x);
                if (label ~= labelsT(i))
                    err = err + 1;
                end
            end
            % Get testing error
            err = err / len;                        
        end
        
        % Get the class label predicted by the group for example x
        function label = getGroupLabel(forest, gnum, x)
            % Get class probabilities
            probs = zeros(size(forest.classes));
            for i=1:length(probs)
                probs(i) = forest.getGroupProbability(x, forest.classes(i), gnum);               
            end
            [~, ind] = max(probs);
            label = forest.classes(ind);
        end
        
    end
    
    
    % Private methods
    methods (Access = private)                
        
        
        
    end
    
    
    % Static methods
    methods (Static)         
        
        % Build a class histogram
        function h = classHist(labels)
            % Build a class histogram                     
            h = zeros(max(labels) + 1, 1);
            for i=1:length(labels)
                lab = labels(i);
                h(lab +1) = h(lab +1) + 1;
            end
        end
        
       
    end
    
    
    
end