classdef GainRandomForest < handle
    %% RANDOMFOREST A class that implements a "random forest" machine learning algorithm.
    %
    % Author: Viachaslau (Slava) Sazonau
    % Project: Implementation and evaluation of Random Forest
    % COMP61011: Machine Learning and Data Mining
    % Date: 12-Octrober-2012
    %
    % Builds and trains a random forest for the given data set.
    % Uses class GainRandomTree.
    
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
        % Selected features
        selFeatures;
    end
    
     % Constants
     properties (Constant)
         % Default number of trees in the forest
         defMaxTrees = 7;
         % Default target class label
         defTargetLabel = 1;
         error_max_size = 'Max size of the forest is exceeded, therefore adding is not possible.';
         error_empty = 'The forest contains no trees. Classification is not possible. Train the forest first.';
         error_outofbounds = 'Tree ID is out of forest bounds. ID must be lower or equal to forest.amount.';
     end
     
     % Public methods
    methods
        
        % Constructor
        function forest = GainRandomForest(data, labels)
                forest.maxTrees = GainRandomForest.defMaxTrees;
                forest.fracSize = GainRandomTree.defFracSize;
                forest.targetLabel = GainRandomForest.defTargetLabel;                
                forest.amount = 0;
                forest.data = data;
                forest.labels = labels;
                forest.classes = unique(forest.labels);
                forest.initSelFeatures();
        end
        
        % Initialise selected features with all features by default
        function initSelFeatures(forest)
            fnum = length(forest.data(1, :));
            forest.selFeatures = zeros(1, fnum);
            for f=1:fnum
                forest.selFeatures(f) = f;
            end            
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
        
        % Get selected features
        function selFeatures = get.selFeatures(forest)
            selFeatures = forest.selFeatures;
        end
        
        % Set selected features
        function set.selFeatures(forest, selFeatures)
            forest.selFeatures = selFeatures;
        end
        
% End of get/set section

        % Get a tree by ID
        function tree = getTree(forest, id)
            if (forest.amount <= 0)
                error(GainRandomForest.error_empty);
            else if (id > forest.amount)
                    error(GainRandomForest.error_outofbounds);
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
                    error(GainRandomForest.error_max_size);
                end
            end
        end
        
        % Grow and train the forest
        function train(forest)            
            for i=1:forest.maxTrees
                indexes = forest.bootstrap();
                tree = GainRandomTree(forest.data, forest.labels);
                tree.fracSize = forest.fracSize;
                tree.targetLabel = forest.targetLabel;
                tree.selFeatures = forest.selFeatures;
                tree.trainIndex(indexes);
                tree.test(forest.data, forest.labels);            
                forest.addTree(tree);
            end            
        end       
     
        % Classify a test example x using the forest
        function label = classify(forest, x)
            if (isempty(forest.first))
                error(GainRandomForest.error_empty);
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
        function label = classifyv00(forest, x)
            if (isempty(forest.first))
                error(GainRandomForest.error_empty);
            else
                % Collect votes
                votes = zeros(forest.amount, 1);
                tree = forest.first;
                for i=1:forest.amount
                    votes(i) = tree.classify(x);
                    tree = tree.next;                    
                end
                % Take a major vote as a class label
                votesHist = GainRandomForest.classHist(votes);
                [~, indVec] = max(votesHist);
                label = indVec(1) - 1;
            end            
        end
        
        % Test the forest classification performance
        function test(forest, dataT, labelsT)
            len = length(labelsT);
            if (len == 0)
                error(GainRandomTree.error_no_input);                
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
        
        % Estimate variable importance
        % Note: (dataT, labelsT) must be an out-of-bag sample for all trees
        function vi = estimateVI(forest, dataT, labelsT)
            fnum = length(forest.selFeatures);
            vi = zeros(1, fnum);
            dataTp =  dataT;            
            for i=1:fnum                
                f = forest.selFeatures(i);
                fvec = dataT(:, f);
                pind = randperm(size(fvec, 1));
                fvecp = fvec(pind);
                % Permute the sample
                dataTp(:, f) = fvecp;
                vif = 0;
                tree = forest.first;
                % Estimate variable importance across trees
                for t=1:forest.amount
                    tree.test(dataT, labelsT);
                    err = tree.err();
                    tree.test(dataTp, labelsT);
                    errp = tree.err();
                    vif = vif + (errp - err);
                    tree = tree.next;
                end                
                vi(i) = vif/forest.amount;
                % Unpermute the sample             
                dataTp(:, f) = fvec;
            end            
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


