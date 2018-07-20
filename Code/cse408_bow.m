% function to create a vocabulary from multiple text files under folders

function feat_vec = cse408_bow(filepath, voc)

[fid, msg] = fopen(filepath, 'rt');
error(msg);
line = fgets(fid); % Get the first line from
 % the file.
feat_vec = zeros(size(voc)); %Initialize the feature vector'

while line ~= -1

    %PUT YOUR IMPLEMENTATION HERE
    words = [];
    punctuations = '[^a-zA-Z\s]';
    line = lower(line);
    line = regexprep(line, punctuations, ' ');
    words = strsplit(line);
    [~, idx] = ismember(words, voc);
    idx = idx(idx > 0);
    line = fgets(fid);
    counts = accumarray(idx', 1)';
    feat_vec = feat_vec + [counts,zeros(1, (length(feat_vec) - length(counts)))];
end
fclose(fid);