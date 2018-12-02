%-----------------------------------------------%
%-------- Practice 1: Feature Extraction -------%
%-----------------------------------------------%

%------------------------------------------------------------------------%
%-------- Part 1: Plotting weight to understand first convolution -------%
%------------------------------------------------------------------------%

% Task - Load the pre-trained Alexnet Network and take a look 
% at its structure using the command ​net.Layers​. 
% Enumerate the different kind of layers and explain them.

% Access the trained model 
net = alexnet;

% See details of the architecture 
disp(net.Layers);


% Task - Plot the weights of the first convolutional layer 
% as it is done in the example and explain intuitively 
% which kind of features do you think that the network is extracting.

% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. 
% There are 96 individual sets of weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')



%-----------------------------------------------------------------%
%-------- Part 2: Train CKDB dataset, and report accuracy  -------%
%-----------------------------------------------------------------%


% -------- Dataset ------%

% Load CKDB dataset 
% Use splitEachLabel method to trim the set.
imds = imageDatastore("./CKDB/", ...
    'IncludeSubfolders',true,...
    'FileExtensions','.tiff',...
    'LabelSource','foldernames');

tbl = countEachLabel(imds);


% -------- Balance dataset ------%
% Balance data in training set
% Because imds above contains an unequal number of images per category, 
% Let's first adjust it, so that the number of images 
% in the training set is balanced. 
% determine the smallest amount of images in a category

%minSetCount = min(tbl{:,2}); 

% Use splitEachLabel method to trim the set.

%imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.

%countEachLabel(imds)


% -------- Split dataset for train and test ------%

% Prepare Training and Test Image Sets
% Below number is ratio between training and testing
% Specify training ratio here.
[trainingSet, testSet] = splitEachLabel(imds, 0.6, 'randomize');
numTrainFiles = size(imds.Files, 1);
% Notice that each set now has exactly the same number of images.
countEachLabel(imds)


% -------- Convert dataset for using Alexnet ------%

% Alexnet was trained by RGB so we also need to 
% convert our grayscale to RGB.
% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');


% -------- Extract training features ------% 
% Typically starting with the layer right before the classification layer 
% is a good place to start.

featureLayer = 'fc8'; % change from tutorial in matlab.
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');


% -------- Train A Multiclass SVM Classifier Using CNN Features ------%
% SVM classifier
% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


% -------- Evaluate Classifier -------- %
% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');


% -------- Accuracy Score ------%
% Get the known labels
testLabels = testSet.Labels;

lenSample = length(testLabels);
cntCorrect = 0;

for i = 1:lenSample
    if testLabels(i) == predictedLabels(i)
        cntCorrect = cntCorrect + 1;
    end
end

fprintf('Accuracy %.2f percent', (cntCorrect/lenSample)*100);
% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);
plotconfusion(testLabels,predictedLabels)
% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
disp(confMat)