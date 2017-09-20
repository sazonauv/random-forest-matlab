%
load heart;

trees = 9;
fearures = 4;

% build a Combinatorial Random Forest
rf1 = ROCRandomForest(data, labels);
rf1.maxTrees = trees;
rf1.fracSize = fearures;
rf1.train();
rf1.trainGroups();
rf1.test(data, labels);
err = rf1.err()
rf1.testByGroups(data, labels);
err = rf1.err()

% repeat = 20;
% tree_ms = zeros(1, repeat);
% otree_ms = zeros(1, repeat);
% tree_er = zeros(1, repeat);
% otree_er = zeros(1, repeat);
% 
% for r=1:repeat
%     r
%     
%     tic;
%     otree = ROCRandomTree(data, labels);
%     otree.train();
%     otree.test(data, labels);
%     otree_ms(r) = round(toc * 1000);
%     otree_er(r) = otree.err();
%     
%     tic;
%     tree = RandomTree(data, labels);
%     tree.train();    
%     tree.test(data, labels);
%     tree_ms(r) = round(toc * 1000);
%     tree_er(r) = tree.err();
%     
% end
% 
% otree_mean = mean(otree_ms)
% tree_mean = mean(tree_ms)
% oerr = mean(otree_er)
% err = mean(tree_er)
% odev = std(otree_er)
% dev = std(tree_er)