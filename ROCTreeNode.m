classdef ROCTreeNode < handle
    %% TREENODE A class that implements a simple tree.
    %
    % Author: Viachaslau (Slava) Sazonau
    % Project: Implementation and evaluation of Random Forest
    % COMP61011: Machine Learning and Data Mining
    % Date: 12-Octrober-2012
    %
    % Implementation uses a simple mechanism of referencing.
    
    % Properties
    properties
        % References
        % left  child
        left;  
        % right child
        right;    
        % parent
        parent;
        % feature for further splitting
        feature;    
        % threshold for further splitting
        threshold;
        % labels stored if it is a leaf, otherwise labels are empty
        label;
        % the number of class instances matched by the condition
        matches;
        % the number of non-class instances matched by the condition
        errors;
    end
    
    % Public methods
    methods
        
        % Constructor
        function node = ROCTreeNode()
            node.label = -1;
        end
      
% Get/set section 
        % Set left child
        function set.left(node, left)
            left.parent = node;
            node.left = left;            
        end
        
        % Get left child
        function left = get.left(node)
            left = node.left;
        end
        
        % Set right child
        function set.right(node, right)
            right.parent = node;
            node.right = right;
        end
        
        % Get right child
        function right = get.right(node) 
            right = node.right;
        end
        
        % Set parent
        function set.parent(node, parent)
            node.parent = parent;
        end
        
        % Get parent
        function parent = get.parent(node)     
            parent = node.parent;
        end
        
        % Set feature
        function set.feature(node, feature)
            node.feature = feature;
        end
        
        % Get feature
        function feature = get.feature(node)
            feature = node.feature;
        end
        
         % Set threshold
        function set.threshold(node, threshold)
            node.threshold = threshold;
        end
        
        % Get threshold
        function threshold = get.threshold(node)
            threshold = node.threshold;
        end
        
        % Get label
        function label = get.label(node)
            label = node.label;
        end
        
        % Set labels
        function set.label(node, label)
            node.label = label;
        end
        
        % Get matches
        function matches = get.matches(node)
            matches = node.matches;
        end
        
        % Set matches
        function set.matches(node, matches)
            node.matches = matches;
        end
        
        % Get errors
        function errors = get.errors(node)
            errors = node.errors;
        end
        
        % Set errors
        function set.errors(node, errors)
            node.errors = errors;
        end
% End of get/set section        
        
    end
    
    
    % Static methods
    methods (Static)
               
        
    end
    
end