classdef GainRandomTree < handle
    %% ROCRANDOMTREE A class that implements a randomized decision tree.
    %
    % Author: Viachaslau (Slava) Sazonau
    % Project: Implementation and evaluation of Random Forest
    % COMP61011: Machine Learning and Data Mining
    % Date: 12-Octrober-2012
    %
    % Builds and trains a randomized decision tree for the given data set.
    % Uses class ROCTreeNode.
    
    % Properties
    properties
        % A root node of type ROCTreeNode
        root;
        % Fraction of features size
        fracSize;
        % Number of thresholds while trying to split
        thrNum;
        % Min possible number of examples in a leaf (leaf size)
        leafSize;
        % Sufficient information gain
        minGain;
        % Reference to the next tree in the forest
        next;
        % Reference to the previous tree in the forest
        previous;
        % Testing error
        testError;
        % Data
        data;
        % Labels
        labels;
        % Data indexes
        indexes;
        % Target class for ROC analysis (i.e. positive)
        targetLabel;
        % Positive points for ROC
        posPoints;
        % Negative points for ROC
        negPoints;        
        % Class histogram for current split (performance consideration)
        classhist;
        % Class labels in the train set
        classes;
        % Handling indexes (performance consideration)
        indexesLeft; 
        indexesRight;
        lengthLeft;
        lengthRight;
        % Selected features
        selFeatures;
    end
    
    % Constants
    properties (Constant)
        % Default fraction size
        defFracSize = 5;
        % Default number of splitting thresholds
        defThrNum  = 3;
        % Default min possible number of examples in a leaf (leaf size)
        defLeafSize = 7;
        % Sufficient information gain (in percentages of max possible)
        defMinGain = 0;
        % Default target class label
        defTargetLabel = 1;
        % Flags for left or right split
        isLeft = 0;
        isRight = 1;
        % Error messages
        error_empty = 'The tree is empty, therefore classification is not possible. Train the tree first.';
        error_no_input = 'No input data. Check your arguments.';
        error_leaf_empty = 'Error occured while training: a leaf is empty.';
    end
    
    % Public methods
    methods
        
        % Constructor
        function tree = GainRandomTree(data, labels)
            tree.root = ROCTreeNode();
            tree.thrNum = GainRandomTree.defThrNum;
            tree.leafSize = GainRandomTree.defLeafSize;
            tree.minGain = GainRandomTree.defMinGain*log2(max(labels)+1);
            tree.data = data;
            tree.labels = labels;
            tree.fracSize = GainRandomTree.defFracSize; %ceil(log2(length(data(1,:))));
            tree.targetLabel = GainRandomTree.defTargetLabel;            
            tree.classes = unique(tree.labels);
            tree.classhist = zeros(length(tree.classes), 1);    
            tree.initSelFeatures();
        end
        
        % Initialise selected features with all features by default
        function initSelFeatures(tree)
            fnum = length(tree.data(1, :));
            tree.selFeatures = zeros(1, fnum);
            for f=1:fnum
                tree.selFeatures(f) = f;
            end            
        end
        
        % Training the tree
        % Uses growtree() which is executed recursively
        function train(tree)
            % Set up indexes for the data and labels
            tree.indexes = zeros(length(tree.labels),1);
            for i=1:length(tree.indexes)
                tree.indexes(i) = i;
            end
            % Grow the tree
            tree.growtree(tree.root, tree.indexes);
        end
        
        % Training the tree using indexes for the data
        % Uses growtree() which is executed recursively
        function trainIndex(tree, indexes)
            % Set up indexes for the data and labels
            tree.indexes = indexes;
            % Grow the tree
            tree.growtree(tree.root, indexes);
        end
        
        
        % Get/set section
        % Get the root
        function root = get.root(tree)
            root = tree.root;
        end
        
        % Get the fraction size
        function fracSize = get.fracSize(tree)
            fracSize = tree.fracSize;
        end
        
        % Set the fraction size
        function set.fracSize(tree, fracSize)
            tree.fracSize = fracSize;
        end
        
        % Get the number of thresholds for splitting
        function thrNum = get.thrNum(tree)
            thrNum = tree.thrNum;
        end
        
        % Set the number of thresholds for splitting
        function set.thrNum(tree, thrNum)
            tree.thrNum = thrNum;
        end
        
        % Get leaf size
        function leafSize = get.leafSize(tree)
            leafSize = tree.leafSize;
        end
        
        % Set leaf size
        function set.leafSize(tree, leafSize)
            tree.leafSize = leafSize;
        end
        
        % Get next tree
        function next = get.next(tree)
            next = tree.next;
        end
        
        % Set next tree
        function set.next(tree, next)
            next.previous = tree;
            tree.next = next;
        end
        
        % Get previous tree
        function previous = get.previous(tree)
            previous = tree.previous;
        end
        
        % Get data
        function data = get.data(tree)
            data = tree.data;
        end
        
        % Set data
        function set.data(tree, data)
            tree.data = data;
        end
        
        % Get labels
        function labels = get.labels(tree)
            labels = tree.labels;
        end
        
        % Set labels
        function  set.labels(tree, labels)
            tree.labels = labels;
        end
        
        % Get target class label
        function targetLabel = get.targetLabel(tree)
            targetLabel = tree.targetLabel;
        end
        
        % Set target class label
        function  set.targetLabel(tree, targetLabel)
            tree.targetLabel = targetLabel;
        end
        
        % Get selected features
        function selFeatures = get.selFeatures(tree)
            selFeatures = tree.selFeatures;
        end
        
        % Set selected features
        function set.selFeatures(tree, selFeatures)
            tree.selFeatures = selFeatures;
        end
        
        % End of get/set section
        
        % Print tree rules
        function printRules(tree)
            GainRandomTree.printNodes(tree.root);
        end
        
        % Get the class probability for example x
        function p = getProbability(tree, x, classLabel)
            if (isempty(tree.root))
                error(GainRandomTree.error_empty);
            else
                p = tree.prob(tree.root, x, classLabel);
            end
        end
        
        % Classify test example x using the tree
        function label = classify(tree, x)
            if (isempty(tree.root))
                error(GainRandomTree.error_empty);
            else
                label = tree.applyRule(tree.root, x);
            end
        end
        
        % Test the tree classification performance
        function test(tree, dataT, labelsT)
            len = length(labelsT);
            if (len == 0)
                error(GainRandomTree.error_no_input);
            end
            err = 0;
            for i=1:len
                x = dataT(i,:);
                label = tree.classify(x);
                if (label ~= labelsT(i))
                    err = err + 1;
                end
            end
            % Get testing error
            err = err / len;
            tree.testError = err;            
        end
        
        % Get testing error
        function testError = err(tree)
            testError = tree.testError;
        end
        
        % Generate ROC curve for the tree
        function curve = generateROC(tree, dataT, labelsT)
            % Find the number of positives and negatives
            positives = 0;
            len = length(labelsT);
            for i=1:len
                if (labelsT(i) == tree.targetLabel)
                    positives = positives + 1;
                end
            end
            negatives = len - positives;
            % Build a ROC curve
            [scores ids] = tree.getScores(dataT, labelsT);            
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
                if (labelsT(ids(i)) == tree.targetLabel)
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
        function [scores ids] = getScores(tree, dataT, labelsT)
            len = length(labelsT);
            scores = zeros(1, len);
            for i=1:len
                scores(i) = tree.getProbability(dataT(i, :), tree.targetLabel);                
            end
            [scores ids] = sort(scores, 'descend');
        end
        
        % Generate ROC curve for the given tree (version 00)
        function curve = generateROCv00(tree)
            % Collect points accross leaves
            tree.descend(tree.root);
            % Compose ROC curve
            positives = 0;           
            len = length(tree.labels);
            for i=1:len
                if (tree.labels(i) == tree.targetLabel)
                    positives = positives + 1;
                end
            end
            negatives = len - positives;
            % Normalise points
            if (~isempty(tree.posPoints) && ~isempty(tree.negPoints))
                for p=1:length(tree.posPoints(:, 1))
                    tree.posPoints(p, 1) = tree.posPoints(p, 1)/negatives;
                    tree.posPoints(p, 2) = tree.posPoints(p, 2)/positives;
                end
                for p=1:length(tree.negPoints(:, 1))
                    tree.negPoints(p, 1) = (negatives - tree.negPoints(p, 1))/negatives;
                    tree.negPoints(p, 2) = (positives - tree.negPoints(p, 2))/positives;
                end
                curve = cat(1, tree.posPoints, tree.negPoints, [0, 0, 0], [1, 1, 1]);
            else
                curve = cat(1, [0, 0, 0], [1, 1, 1]);
            end
            curve = sort(curve);
        end
        
    end
    
    % Private methods
    methods (Access = private)
        
        % Grow a tree from the current node (recursion!)
        function growtree(tree, node, indexes)
            % Obtain matches and errors for a class     
            node.label = tree.maxLabel(indexes);
            len = length(indexes);
            matches = len;
            errors = 0;
            for i=1:len
                if (tree.labels(indexes(i)) ~= node.label)                
                    errors = errors + 1;
                end
            end
            % Set matches and errors for the node
            node.matches = matches;
            node.errors = errors;            
            % Leaf size is reached
            if (length(indexes) > tree.leafSize)            
                numFe = length(tree.selFeatures);
                % Select the fraction of features randomly
                fraction = randperm(numFe, tree.fracSize);
                maxfGain = 0;
                fmax = 0;
                tmax = 0;
                % Find max and min
                xmax = -realmax();
                xmin = realmax();
                for i=1:length(fraction)
                    for ind=1:length(indexes)
                        val = tree.data(indexes(ind), tree.selFeatures(fraction(i)));
                        if (val > xmax)
                            xmax = val;
                        end
                        if (val < xmin)
                            xmin = val;
                        end
                    end
                end
                % Generate random thresholds % ceil(log2(length(indexes)))
                randvec = rand(tree.thrNum, 1);
                tvec = double(xmin) + randvec.*double(xmax - xmin);
                H = tree.entropy(indexes);
                for i=1:length(fraction)
                    f = tree.selFeatures(fraction(i));
                    % Find best feature and threshold
                    tfmax = 0;
                    maxtGain = 0;
                    for j=1:length(tvec)
                        t = tvec(j);
                        % Calculate information gain
                        infGain = tree.infgain(indexes, f, t, H);
                        % Find max gain for a threshold t
                        if (maxtGain <= infGain)
                            maxtGain = infGain;
                            tfmax = t;
                        end
                    end
                    % Find max gain for a feature f
                    if (maxfGain <= maxtGain)
                        maxfGain = maxtGain;
                        fmax = f;
                        tmax = tfmax;
                    end
                end
                
                % If best gain is not sufficient, make this node as a leaf (no pruning for Random Forest!)
                if (maxfGain > tree.minGain)                    
                    % Split the node
                    [indLeft, indRight] = tree.splitData(indexes, fmax, tmax);
                    % Create childs
                    left = ROCTreeNode();
                    right = ROCTreeNode();
                    % Add childs
                    node.left = left;
                    node.right = right;
                    % Set feature
                    node.feature = fmax;
                    % Set threshold
                    node.threshold = tmax;
                    % Carry out recursion
                    tree.growtree(left, indLeft);
                    tree.growtree(right, indRight);
                end
            end
        end
                
        % Apply classification rule on the given node
        function label = applyRule(tree, node, x)
            % A leaf is reached
            if (isempty(node.left))
                if (node.label < 0)
                    error(GainRandomTree.error_leaf_empty);
                else
                    label = node.label;
                end
                return;
            end
            % Apply rule
            if (x(node.feature) < node.threshold)
                label = tree.applyRule(node.left, x);
            else
                label = tree.applyRule(node.right, x);
            end
        end
            
        % Get most probable label from the histogram
        function label = maxLabel(tree, indexes)
            h = tree.classHist(indexes);
            maxLabel = 0;
            maxNum = 0;
            for i=1:length(h)
                num = h(i);
                if (maxNum < num)
                    maxNum = num;
                    maxLabel = tree.classes(i);
                end
            end
            label = maxLabel;
        end
        
        % Calculate information gain
        function infGain = infgain(tree, indexes, f, t, H)
            % Calculate H(labels)
            if (length(indexes) <= 1)                
                infGain = 0;
                return;
            end
            % Split labels
            tree.checkSplit(indexes, f, t);
            % Calculate H(labels|left)
            if (tree.lengthLeft > 1)
                Hl = tree.checkEntropy(GainRandomTree.isLeft);
            else
                % Hl == 0; Hr == H;
                infGain = 0;
                return;
            end
            % Calculate H(labels|right)
            if (tree.lengthRight > 1)
                Hr = tree.checkEntropy(GainRandomTree.isRight);
            else
                % Hr == 0; Hl == H;
                infGain = 0;
                return;
            end
            % Finally calculate information gain for the given split
            infGain = H - ( tree.lengthLeft/length(indexes)*Hl + tree.lengthRight/length(indexes)*Hr);
        end
        
        % Calculate entropy
        function e = checkEntropy(tree, splitFlag)
            % Build a class histogram
            tree.checkClassHist(splitFlag);
            e = 0;
            if (splitFlag == GainRandomTree.isLeft)              
               for i=1:length(tree.classhist)
                    if (tree.classhist(i) > 0)
                        e = e - ( tree.classhist(i)/tree.lengthLeft )*log2( tree.classhist(i)/tree.lengthLeft );
                    end
                end
            else                
                for i=1:length(tree.classhist)
                    if (tree.classhist(i) > 0)
                        e = e - ( tree.classhist(i)/tree.lengthRight )*log2( tree.classhist(i)/tree.lengthRight );
                    end
                end
            end            
        end
        
        % Calculate entropy
        function e = entropy(tree, indexes)
            % Build a class histogram
            h = tree.classHist(indexes);
            % Calculate entropy
            e = 0;
            for i=1:length(h)
                if (h(i) > 0)
                    e = e - ( h(i)/length(indexes) )*log2( h(i)/length(indexes) );
                end
            end
        end
        
        % Split data and labels (performance consideration)
        function checkSplit(tree, indexes, f, t)
            % Evaluate left and right split size
            tree.lengthLeft = 0;
            tree.lengthRight = 0;
            for i=1:length(indexes)
                if (tree.data(indexes(i), f) < t)
                    tree.lengthLeft = tree.lengthLeft + 1;
                else
                    tree.lengthRight = tree.lengthRight +1;
                end
            end            
            il = 1;
            ir = 1;
            for i=1:length(indexes)
                if (tree.data(indexes(i), f) < t)
                    tree.indexesLeft(il) = indexes(i);
                    il = il + 1;
                else
                    tree.indexesRight(ir) = indexes(i);
                    ir = ir + 1;
                end
            end
        end
        
        % Split data and labels (performance consideration)
        function [indexesLeft, indexesRight] = splitData(tree, indexes, f, t)
            % Evaluate left and right split size
            lenLeft = 0;
            lenRight = 0;
            for i=1:length(indexes)
                if (tree.data(indexes(i), f) < t)
                    lenLeft = lenLeft + 1;
                else
                    lenRight = lenRight +1;
                end
            end
            % Get subsamples
            indexesLeft = zeros(lenLeft, 1);
            indexesRight = zeros(lenRight, 1);
            il = 1;
            ir = 1;
            for i=1:length(indexes)
                if (tree.data(indexes(i), f) < t)
                    indexesLeft(il) = indexes(i);
                    il = il + 1;
                else
                    indexesRight(ir) = indexes(i);
                    ir = ir + 1;
                end
            end
        end
        
        % Build a class histogram
        function checkClassHist(tree, splitFlag)
            tree.classhist(tree.classhist>0) = 0;
            if (splitFlag == GainRandomTree.isLeft)
                for i=1:tree.lengthLeft
                    lab = tree.labels(tree.indexesLeft(i));
                    ind = find(tree.classes==lab);
                    tree.classhist(ind) = tree.classhist(ind) + 1;
                end                
            else               
                for i=1:tree.lengthRight
                    lab = tree.labels(tree.indexesRight(i));
                    ind = find(tree.classes==lab);
                    tree.classhist(ind) = tree.classhist(ind) + 1;
                end
            end
        end
        
        % Build a class histogram
        function h = classHist(tree, indexes)
            % Build a class histogram               
            h = zeros(size(tree.classes));
            for i=1:length(indexes)
                lab = tree.labels(indexes(i));
                ind = find(tree.classes==lab);
                h(ind) = h(ind) + 1;
            end
        end
        
        % Recursive function on nodes
        function descend(tree, node)
            % It is a leaf
            if (isempty(node.left))
                TP = node.matches - node.errors;
                FP = node.errors;
                if (node.label == tree.targetLabel)                    
                    score = FP/(node.matches*(1+exp(-node.matches)));
                    pt = [FP, TP, score];
                    tree.posPoints = cat(1, tree.posPoints, pt);
                else
                    score = TP/(node.matches*(1+exp(-node.matches)));
                    pt = [TP, FP, score];
                    tree.negPoints = cat(1, tree.negPoints, pt);
                end                
            else
                tree.descend(node.left);
                tree.descend(node.right);
            end
            
        end
        
        % Recursive function to get the class probability for x
        function p = prob(tree, node, x, classLabel)
            % A leaf is reached
            if (isempty(node.left))                
                if (node.label == classLabel)                  
                    p = (node.matches - node.errors)/(node.matches*(1+exp(-node.matches)));
                else
                    p = node.errors/(node.matches*(1+exp(-node.matches)));
                end                
                return;
            end
            % Apply rule
            if (x(node.feature) < node.threshold)
                p = tree.prob(node.left, x, classLabel);
            else
                p = tree.prob(node.right, x, classLabel);
            end
        end
        
    end
    
    % Static methods
    methods (Static)
        
        % Threshold averaging of ROC curves
        function [averCurve, deviations] = averageThresholdROC(curves, samples)
            nrocs = length(curves);
            T = [];
            for i=1:nrocs                
                curve = curves{i};
                [npts ~] = size(curve);
                curveT = zeros(1, npts);
                for j=1:npts
                    curveT(j) = curve(j, 3);
                end
                T = cat(2, T, curveT);    
            end            
            T = sort(T, 'descend');
            averCurve = zeros(samples, 2);
            deviations = zeros(samples, 2);
            z95 = 1.96;
            step = floor(length(T)/samples);
            for s=1:samples
                fprs = zeros(1, nrocs);
                tprs = zeros(1, nrocs);
                for i=1:nrocs
                    curve = curves{i};
                    [npts ~] = size(curve);
                    [fpr, tpr] = GainRandomTree.rocPointAtThr(curve, npts, T((s-1)*step + 1));
                    fprs(i) = fpr;
                    tprs(i) = tpr;
                end
                averCurve(s, 1) = mean(fprs);
                averCurve(s, 2) = mean(tprs);
                deviations(s, 1) = std(fprs)*(z95/sqrt(nrocs));
                deviations(s, 2) = std(tprs)*(z95/sqrt(nrocs));
            end
            averCurve = cat(1, averCurve, [1, 1]);
            deviations = cat(1, deviations, [0, 0]);
        end
        
        % Get ROC points at given threshold
        function [fpr, tpr] = rocPointAtThr(curve, npts, t)
            i = 1;
            while (i<npts && curve(i, 3)>t)
                i = i + 1;
            end
            fpr = curve(i, 1);
            tpr = curve(i, 2);
        end
        
        % Print nodes
        function printNodes(node)
            if (isempty(node))
                return;
            end
            f = node.feature;
            t = node.threshold;
            l = node.label;
            m = node.matches;
            e = node.errors;
            str = sprintf(' x%u < %3.4f : %2.0f  = %2.0f, %2.0f', f, t, l, m, e);
            disp(str);
            GainRandomTree.printNodes(node.left);
            GainRandomTree.printNodes(node.right);
        end
        
    end
    
end