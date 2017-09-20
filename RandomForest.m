classdef RandomForest < handle
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
        % Class labels in the train set
        classes;
    end
    
     % Constants
    properties (Constant)
        % Default number of trees in the forest
        defMaxTrees = 11;        
        % Error messages
        error_max_size = 'Max size of the forest is exceeded, therefore adding is not possible.';
        error_empty = 'The forest contains no trees. Classification is not possible. Train the forest first.';
    end
    
    % Public methods
    methods
        
        % Constructor
        function forest = RandomForest(data, labels)
                forest.maxTrees = RandomForest.defMaxTrees;
                forest.fracSize = ROCRandomTree.defFracSize;
                forest.amount = 0;
                forest.data = data;
                forest.labels = labels;
                forest.targetLabel = ROCRandomForest.defTargetLabel;
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
                    error(RandomForest.error_max_size);
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
                forest.addTree(tree);
            end            
        end
        
        % Classify a test example x using the forest
        function label = classify(forest, x)
            if (isempty(forest.first))
                error(RandomForest.error_empty);
            else
                % Collect votes
                votes = zeros(forest.amount, 1);
                tree = forest.first;
                i = 1;
                while(i <= forest.amount && ~isempty(tree))                    
                    votes(i) = tree.classify(x);
                    tree = tree.next;
                    i = i + 1;
                end
                % Take a major vote as a class label
                votesHist = RandomForest.classHist(votes);
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
                % Make all weights identical (simple majority voting scheme)
                weights(i) = 1;
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

