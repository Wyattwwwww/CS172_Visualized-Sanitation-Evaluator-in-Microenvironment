function [training_set, test_set] = loadAndSplit(path,train_size)
    % load dataset
    imds = imageDatastore(path,'IncludeSubfolders',true,'LabelSource',...
        'foldernames'); 
    % split the images into training and test data
    [training_set,test_set] = splitEachLabel(imds,train_size,'randomize');
end