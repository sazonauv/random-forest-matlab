classdef RandomPlane < handle
    %% RANDOMTREE A class that implements a randomized decision tree.
    %
    % Author: Viachaslau (Slava) Sazonau
    % Project: Implementation and evaluation of Random Forest
    % COMP61011: Machine Learning and Data Mining
    % Date: 12-Octrober-2012
    %
    % Builds and trains a randomized decision tree for the given data set.
    % Uses class PlaneNode.
    
    % Properties
    properties        
        % A root node of type PlaneNode
        root;
        % Fraction of features size
        fracSize;
        % Number of random weight vectors on each split
        weighNum;
        % Number of thresholds while trying to split
        thrNum;
        % Min possible number of examples in a leaf (leaf size)
        leafSize;        
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
    end
    
    % Constants
    properties (Constant)
        % Default fraction size
        defFracSize = 5;
        % Default number weight vectors on each split    
        defWeighNum = 10;           
        % Default number of splitting thresholds         
        defThrNum  = 10;
        % Default min possible number of examples in a leaf (leaf size)
        defLeafSize = 5;        
        % Error messages
        error_empty = 'The tree is empty, therefore classification is not possible. Train the tree first.';
        error_no_input = 'No input data. Check your arguments.';
        error_leaf_empty = 'Error occured while training: a leaf is empty.';
    end
    
    % Public methods
    methods
        
        % Constructor
        function tree = RandomPlane(data, labels) 
                tree.root = PlaneNode();
                tree.fracSize = RandomPlane.defFracSize;
                tree.weighNum = RandomPlane.defWeighNum;
                tree.thrNum = RandomPlane.defThrNum;
                tree.leafSize = RandomPlane.defLeafSize;                
                tree.data = data;
                tree.labels = labels;                
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
        
        % Get the number of weight vectors
        function weighNum = get.weighNum(tree)
            weighNum = tree.weighNum;
        end
        
        % Set the number of weight vectors
        function set.weighNum(tree, weighNum)
            tree.weighNum = weighNum;
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
        
% End of get/set section

        % Print tree rules
        function printRules(tree)
           RandomPlane.printNodes(tree.root); 
        end
        
        % Classify test example x using the tree
        function label = classify(tree, x)
            if (isempty(tree.root))
                error(RandomPlane.error_empty);
            else
                label = tree.applyRule(tree.root, x);
            end
        end
        
        % Test the tree classification performance
        function test(tree, dataT, labelsT)
            len = length(labelsT);
            if (len == 0)
                error(RandomPlane.error_no_input);                
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
        
    end
    
    % Private methods
    methods (Access = private)
        
        % Grow a tree from the current node (recursion!)
        function growtree(tree, node, indexes)
            if (length(indexes) <= tree.leafSize)
                node.label = tree.maxLabel(indexes);
            else                
                numFe = length(tree.data(1,:));                
                
               % Select the fraction of features randomly
                fraction = randperm(numFe, tree.fracSize);                
                
                [wmax, tmax, maxGain] = tree.gradient(indexes, fraction);                
                
                % If best gain is not sufficient, make this node as a leaf (no pruning for Random Forest!)
                if (maxGain <= 0)
                    node.label = tree.maxLabel(indexes);
                else
                    % Split the node
                    [indexesLeft, indexesRight] = tree.splitData(indexes, fraction, wmax, tmax);
                    % Create childs
                    left = PlaneNode();
                    right = PlaneNode();
                    % Add childs
                    node.left = left;
                    node.right = right;
                    % Set feature
                    node.features = fraction;
                    % Set weights
                    node.weights = wmax;
                    % Set threshold
                    node.threshold = tmax;
                    % Carry out recursion
                    tree.growtree(left, indexesLeft);
                    tree.growtree(right, indexesRight);
                end
            end            
        end
        
        % Gradient descent
        function [wmax, tmax, maxGain] = gradient(tree, indexes, fraction)           
            % Generate random thresholds
            xmax = -realmax();
            xmin = realmax();
            for f=1:length(fraction)
                for ind=1:length(indexes)
                    val = tree.data(indexes(ind), fraction(f));
                    if (val > xmax)
                        xmax = val;
                    end
                    if (val < xmin)
                        xmin = val;
                    end
                end
            end
            t = xmin + (xmax - xmin)*rand(1, 1);
            % Generate random weights
            w = rand([1, length(fraction)]);
            w = RandomPlane.normalise(w);
            % Set alpha and epochs
            alpha = 0.0001;
            epochs = length(fraction);
            gain = 0;
            % Find maximum
            for e=1:epochs
                newgain = tree.infgain(indexes, fraction, w, t);
                 if (newgain ~= 0 || gain ~= 0)
                    for i=1:length(indexes)
                        for f=1:length(fraction)
                            w(f) = w(f) - alpha*(newgain - gain)*tree.data(indexes(i), fraction(f));
                        end
                    end
                    t = t + alpha*(newgain - gain);
                    gain = newgain;
                 end
            end
            maxGain = gain;
            wmax = w;
            tmax = t;
        end
        
        % Apply classification rule on the given node
        function label = applyRule(tree, node, x)
            % A leaf is reached
            if (isempty(node.left))
                if (node.label < 0)
                    error(RandomPlane.error_leaf_empty);
                else               
                    label = node.label;
                end
                return;
            end
            % Apply rule
            if ( RandomPlane.splitFunction(x, node.features, node.weights, node.threshold) < 0.5)
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
                    maxLabel = i - 1;
                end
            end
            label = maxLabel;
        end
        
        % Calculate information gain
        function infGain = infgain(tree, indexes, fraction, w, t)            
            % Calculate H(labels)
            if (length(indexes) > 1)
                H = tree.entropy(indexes);
            else
                infGain = 0;
                return;
            end
            % Split labels
            [indexesLeft, indexesRight] = tree.splitData(indexes, fraction, w, t);
            % Calculate H(labels|left)
            if (length(indexesLeft) > 1)
                Hl = tree.entropy(indexesLeft);
            else
                % Hl == 0; Hr == H;
                infGain = 0;
                return;
            end
            % Calculate H(labels|right)
            if (length(indexesRight) > 1)
                Hr = tree.entropy(indexesRight);
            else
               % Hr == 0; Hl == H;
                infGain = 0;
                return;
            end
            % Finally calculate information gain for the given split
            infGain = H - ( length(indexesLeft)/length(indexes)*Hl + length(indexesRight)/length(indexes)*Hr);            
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
        function [indexesLeft, indexesRight] = splitData(tree, indexes, fraction, weights, t)
            % Evaluate left and right split size                        
            lengthLeft = 0;
            lengthRight = 0;            
            for i=1:length(indexes)
                val = RandomPlane.splitFunction(tree.data(indexes(i),:), fraction, weights, t);
                if (val < 0.5)
                    lengthLeft = lengthLeft + 1;
                else
                    lengthRight = lengthRight +1;
                end
            end
            % Get subsamples
            indexesLeft = zeros(lengthLeft, 1);            
            indexesRight = zeros(lengthRight, 1);            
            il = 1;
            ir = 1;
            for i=1:length(indexes)
                val = RandomPlane.splitFunction(tree.data(indexes(i),:), fraction, weights, t);
                if (val < 0.5)
                    indexesLeft(il) = indexes(i);                    
                    il = il + 1;
                else
                    indexesRight(ir) = indexes(i);                    
                    ir = ir + 1;
                end
            end
        end
        
        % Build a class histogram
        function h = classHist(tree, indexes)
            % Build a class histogram
            lmax = -realmax();
            for i=1:length(indexes)
                lab = tree.labels(indexes(i));
                if (lmax < lab)
                    lmax = lab;
                end
            end            
            h = zeros(lmax + 1, 1);
            for i=1:length(indexes)
                lab = tree.labels(indexes(i));
                h(lab +1) = h(lab +1) + 1;
            end
        end
        
    end
    
    % Static methods
    methods (Static)        
        % Normalise weights
        function weights = normalise(weights)
            [W, ~] = size(weights);
            for w=1:W
                weights(w,:) = weights(w,:) / norm(weights(w,:));
            end
        end
        
        % Split function
        function val = splitFunction(x, fraction, weights, threshold)
            val = 0;
            for f=1:length(fraction)
                val = val + weights(f)*x(fraction(f));
            end
            val = 1/(1+exp(threshold - val));
        end
        
        % Print nodes
        function printNodes(node)
            if (isempty(node))
                return;
            end            
            t = node.threshold;
            l = node.label;
            str = sprintf(' x < %2.2f : %2.0f ', t, l);
            disp(str);
            RandomPlane.printNodes(node.left);
            RandomPlane.printNodes(node.right);
        end
        
    end
    
end

