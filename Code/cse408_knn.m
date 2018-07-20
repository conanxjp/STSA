% function to run KNN classification


function pred_label = cse408_knn(test_feat_set, train_label, train_feat_set, k, DstType)


if DstType == 1 %SSD
    %PUT YOUR CODE HERE
    dist = pdist2(test_feat_set', train_feat_set', 'squaredeuclidean');
%     cellArray = num2cell(train_feat_set);
%     dist = cellfun(@(x) sqrt(sum((x - test_feat_set).^2)), cellArray, 'Uni', false);
%     dist = cell2mat(dist);
elseif DstType == 2 %Angle Between Vectors
    %PUT YOUR CODE HERE 
    dist = pdist2(test_feat_set', train_feat_set', 'cosine');
elseif DstType == 3 %Number of words in common
    %PUT YOUR CODE HERE
    dist = [];
    for i = 1:size(train_feat_set,2)
        dist = [dist, sum(min([test_feat_set,train_feat_set(:,i)]'))];
    end
    dist = -dist; % Why minus? Because number of words in common is similarity
end



%Find the top k nearest neighbors, and do the voting. 

[B,I] = sort(dist);

posCt=0;
negCt=0;
for ii = 1:k
    if train_label(I(ii)) == 1
        posCt = posCt + 1;
    elseif train_label(I(ii)) == 0
        negCt = negCt + 1;
    end    
end

if posCt >= negCt
    pred_label = 1;
else
    pred_label = 0;
end