%% Mode 0: Use models 
mode = 1;
data_path = '7_types/train'; % change your dataset path here

if mode == 1
    %% Mode 1: Use models after new training by BoW
    is_resized = 1;
    
    voca = 1000;
    features = 0.7;
    extractorFcn = @fastBagOfFeaturesExtractor; % set extractor here
    
    t1 = clock;
    train_size = 500;

    [training_set, test_set] = loadAndSplit(data_path, train_size);
    bag = bagOfFeatures(training_set,'CustomExtractor',extractorFcn,...
    'VocabularySize',voca,'StrongestFeatures',features);
    category_classifier = trainImageCategoryClassifier(training_set,bag);

    t2 = clock;
    t = etime(t2,t1);

    conf_matrix = evaluate(category_classifier,test_set);

    result_name = sprintf("%d_BoW_%d_%d_0%d", is_resized, train_size, ...
        voca, features*10);
    save(['models\', convertStringsToChars(result_name)])

    clearvars training_set test_set;
    clearvars bag category_classifier conf_matrix;

    
elseif mode == 2
    %% Mode 2: Use models after new training by SPM
    is_resized = 1;
    
    voca = 1500;
    features = 0.8;
    extractorFcn = @superSpatialPyramidExtractor; 
  
    t1 = clock;
    train_size = 2000;

    [training_set, test_set] = loadAndSplit(data_path, train_size);
    bag = bagOfFeatures(training_set,'CustomExtractor',extractorFcn,...
    'VocabularySize',voca,'StrongestFeatures',features);
    category_classifier = trainImageCategoryClassifier(training_set,bag);

    t2 = clock;
    t = etime(t2,t1);

    conf_matrix = evaluate(category_classifier,test_set);

    result_name = sprintf("%d_SPM_%d_%d_0%d", is_resized, train_size, ...
        voca, features*10);
    save(['models\', convertStringsToChars(result_name)])

    clearvars training_set test_set;
    clearvars bag category_classifier conf_matrix;

end
% You can use the model in the following way:
% img = imread('MyOwnTest\digger.jpg');
% [labelIdx, score] = predict(category_classifier,img);
    