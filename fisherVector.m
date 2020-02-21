clear all;
% First set vlfeat if not set
run('C:\Users\asus\vlfeat\vlfeat-0.9.21\toolbox\vl_setup.m'); 

disp('Divide');
% Read in data
TrainDir=('small_dataset\train');
TestDir=('small_dataset\test');
TrainSet = imageSet(TrainDir,'recursive');
TestSet = imageSet(TestDir,'recursive');
classes = size(TrainSet,2);
trainnum = TrainSet(1,1).Count;
testnum = TestSet(1,1).Count;

resize = 250;

disp('SIFT to Fisher vectors');
% Get features
Train_Fisher = cell(1,classes*trainnum);
Test_Fisher = cell(1,classes*testnum);
for i = 1:classes
    for j = 1:trainnum
        train_name = TrainSet(1,i).ImageLocation{1,j};
        train_img = imresize(imread(train_name),[NaN resize]);
        if size(train_img,3) == 3
            train_img = single(rgb2gray(train_img));
        else
            train_img = single(train_img);
        end
        [~,discriptors] = vl_sift(train_img);
        K = 3;
        [means, covariances, priors] = vl_gmm(double(discriptors), K);
        Train_Fisher{1,(i-1)*trainnum+j} = vl_fisher(double(discriptors), ...
            means, covariances, priors, 'normalized');
    end    
end

maxlen = 0;
for i = 1:classes
    for j = 1:testnum
        test_name = TestSet(1,i).ImageLocation{1,j};
        test_img = imresize(imread(test_name),[NaN resize]);
        if size(test_img,3) == 3
            test_img = single(rgb2gray(test_img));
        else
            test_img = single(test_img);
        end
        [~,discriptors] = vl_sift(test_img);
        K = 3;
        [means, covariances, priors] = vl_gmm(double(discriptors), K);
        Test_Fisher{1,(i-1)*testnum+j} = vl_fisher(double(discriptors),... 
            means, covariances, priors,'normalized');
        maxlen = max(maxlen, length(Test_Fisher{1,(i-1)*testnum+j}));
    end    
end

disp('SVM');
% SVM
TrainLabels = zeros(classes*trainnum,1);
TrainMatrix = zeros(classes*trainnum,maxlen);
for i = 1:classes
    for j = 1:trainnum
        TrainLabels((i-1)*trainnum+j,1) = double(i);
        TrainMatrix((i-1)*trainnum+j,:) = Train_Fisher{1,(i-1)*trainnum+j}(:,1);
    end
end
Classifier = fitcecoc(TrainMatrix,TrainLabels); 

disp('Predict');
% Predict
TestLabels = zeros(classes*testnum,1);
TestMatrix = zeros(classes*testnum,maxlen);
for i = 1:classes
    for j = 1:testnum
        TestLabels((i-1)*testnum+j,1) = double(i);
        TestMatrix((i-1)*testnum+j,:) = Test_Fisher{1,(i-1)*testnum+j}(:,1);
    end
end

[label, score] = predict(Classifier, TestMatrix);
count = 0;
for i = 1:length(label)
    if label(i) == TestLabels(i)
        count = count + 1;
    end
end
accuracy = count/length(label)

result_name = sprintf("Fisher_K%d_R%d", K, resize);
save(['models\', convertStringsToChars(result_name)])











