function label = classifier(path)
    load models\1_BoW_500_1000_07 category_classifier
    img = imread(path);
	[idx, ~] = predict(category_classifier,img);
    label = category_classifier.Labels{idx};
end