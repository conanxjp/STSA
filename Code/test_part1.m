accuracy_all = zeros([6,3,8]);
threshold = [1,2,3,5,7,10,15,20];
for i = 1:8
    accuracy_all(:,:,i) = part1_func(threshold(i));
end