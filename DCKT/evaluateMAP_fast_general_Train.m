% function [map, mapCategory] = evaluateMAP_fast_general(queryResult,queryCat,testCat,catNum, quaryCatDiff)
function [map, mapCategory] = evaluateMAP_fast_general_Train(queryResult,queryCat,testCat,catNum)

resFlg = testCat(queryResult);

%tic
map = zeros(length(queryCat),1);

for i = 1:size(queryResult,1)
    query = resFlg(i,:);
    d = find(query==queryCat(i));
    d = (1:length(d))./d(1:end);
    map(i) = mean(d);
end

Category = unique(queryCat);
for i = 1:length(Category)
%     mapCategory(i,1) = mean(ap(queryCat == Category(i)));
      mapCategory(i,1) = 0;
end

end