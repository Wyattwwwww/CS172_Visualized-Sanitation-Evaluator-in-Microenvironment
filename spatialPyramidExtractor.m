function [features, featureMetrics] = spatialPyramidExtractor(I)
% This function implements the default SURF feature extraction used in
% bagOfFeatures and is only intended to show how to write a custom 
% extractor function for bagOfFeatures.
%
% [features, featureMetrics] = exampleBagOfFeaturesExtractor(I) returns
% SURF features extracted over a regular grid of point locations at
% multiple scales.
%
% [..., featureLocations] = exampleBagOfFeaturesExtractor(I) optionally
% return the feature locations. This is used by the indexImages function
% for creating a searchable image index.
%
% Example: Using custom features in bagOfFeatures
% ------------------------------------------------
% % Define a set of images
% setDir = fullfile(toolboxdir('vision'),'visiondata','imageSets');
% imgDs = imageDatastore(setDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% 
% % Specify a custom extractor function
% extractor = @exampleBagOfFeaturesExtractor;
% customBag = bagOfFeatures(imgDs, 'CustomExtractor', extractor)
%
% See also bagOfFeatures, retrieveImages, indexImages
 
%% Step 1: Preprocess the Image
% The extractor function is applied to each image, I, within the image set
% used to create the bagOfFeatures. Depending on the type of features being
% extracted, the input images may require preprocessing prior to feature
% extraction. For SURF features, I must be a grayscale image.

% Convert I to grayscale if required.
[height,width,numChannels] = size(I);
if numChannels > 1
    grayImage = rgb2gray(I);
else
    grayImage = I;
end

% Split the image into 4 subgraphs to have spatial pyramid infos.
half_h = floor(height/2);
half_w = floor(width/2);

grayImage_1 = grayImage(1:half_h, 1:half_w);
grayImage_2 = grayImage(1:half_h, (half_w+1):width);
grayImage_3 = grayImage((half_h+1):height, 1:half_w);
grayImage_4 = grayImage((half_h+1):height, (half_w+1):width);

%% Step 2: Select Point Locations for Feature Extraction
% Use a feature detector such as detectSURFFeatures
% or detectMSERFeatures to select point locations. 
multiscaleSURFPoints_1 = detectSURFFeatures(grayImage_1);
multiscaleSURFPoints_2 = detectSURFFeatures(grayImage_2);
multiscaleSURFPoints_3 = detectSURFFeatures(grayImage_3);
multiscaleSURFPoints_4 = detectSURFFeatures(grayImage_4);
                    
%% Step 3: Extract features
% Finally, extract features from the selected point locations. By default,
% bagOfFeatures extracts upright SURF features. 
features_1 = extractFeatures(grayImage_1, multiscaleSURFPoints_1,'Upright',true);
features_2 = extractFeatures(grayImage_2, multiscaleSURFPoints_2,'Upright',true);
features_3 = extractFeatures(grayImage_3, multiscaleSURFPoints_3,'Upright',true);
features_4 = extractFeatures(grayImage_4, multiscaleSURFPoints_4,'Upright',true);
features = [features_1; features_2; features_3; features_4];

%% Step 4: Compute the Feature Metric
% The feature metrics indicate the strength of each feature, where larger
% metric values are given to stronger features. The feature metrics are
% used to remove weak features before bagOfFeatures learns a visual
% vocabulary. You may use any metric that is suitable for your feature
% vectors.
%
% Use the variance of the SURF features as the feature metric.
featureMetrics = var(features,[],2);

% Alternatively, if a feature detector was used for point selection,
% the detection metric can be used. For example:
%
% featureMetrics = multiscaleSURFPoints.Metric;

