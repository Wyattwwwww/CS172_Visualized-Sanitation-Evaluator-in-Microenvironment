function [features, featureMetrics, varargout] = bagOfFeaturesExtractor(I)
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
numChannels = size(I,3);
if numChannels > 1
    grayImage = rgb2gray(I);
else
    grayImage = I;
end

%% Step 2: Select Point Locations for Feature Extraction
% Here, a regular spaced grid of point locations is created over I. This
% allows for dense SURF feature extraction. 
                  
% Use a feature detector such as detectSURFFeatures
% or detectMSERFeatures to select point locations.
multiscaleSURFPoints = detectSURFFeatures(grayImage);
                    
%% Step 3: Extract features
% Finally, extract features from the selected point locations. By default,
% bagOfFeatures extracts upright SURF features. 
features = extractFeatures(grayImage, multiscaleSURFPoints,'Upright',true);

%% Step 4: Compute the Feature Metric
% The feature metrics indicate the strength of each feature, where larger
% metric values are given to stronger features. The feature metrics are
% used to remove weak features before bagOfFeatures learns a visual
% vocabulary. You may use any metric that is suitable for your feature
% vectors.
featureMetrics = multiscaleSURFPoints.Metric;

% Optionally return the feature location information. The feature location
% information is used for image search applications. See the retrieveImages
% and indexImages functions.
if nargout > 2
    % Return feature location information
    varargout{1} = multiscaleSURFPoints.Location;
end


