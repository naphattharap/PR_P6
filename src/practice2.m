%-----------------------------------------------%
%-------- Practice 2: CNN Classifier -----------%
%-----------------------------------------------%


% The network architecture can vary depending on the types
% and numbers of layers included and depends on the particular application or data

% For example, if you have categorical responses,
% you must have a softmax layer and a classification layer.
% whereas if your response is continuous,
% you must have a regression layer at the end of the network.

% The number of weights in a filter is h * w * c,
% c is the number of channels in the input.
% This number corresponds to the number of neurons
% in the convolutional layer that connect to the same region
% in the input.


% Stride:
% Step size for traversing the input vertically and horizontally,
% specified as a vector of two positive integers [a b],
% where a is the vertical step size and b is the horizontal step size.
% When creating the layer, you can specify Stride as a scalar
% to use the same value for both dimensions.
% If the stride dimensions are less than
% the respective pooling dimensions, then the pooling regions overlap.


% A smaller network with only one or two convolutional layers
% might be sufficient to learn on a small number of grayscale image data.
% On the other hand, for more complex data with millions of colored images,
% you might need a more complicated network with multiple convolutional
% and fully connected layers.

% Load dataset
imds = imageDatastore("./CKDB/", ...
    'IncludeSubfolders',true,'FileExtensions','.tiff','LabelSource','foldernames');

% Check dataset
labelCount = countEachLabel(imds);
img = readimage(imds,1);
size(img)


% Define the network architecture.

%------ Architecture Design ------%

% A classic CNN architecture
% Input --> Conv --> ReLu --> Conv --> ReLu --> Pool --> ReLu
% --> Conv --> ReLu --> Pool --> Fully Connected.

% - Image input layer.
% It takes input image as 128x128 as for CKDB dataset's images.
% and 1 for numChannel of grayscale image.
% - The first layer in a CNN is always a Convolutional Layer.
% Set filter (or sometimes referred to as a neuron or a kernel)
% which is an array of number (the numbers are called weights or parameters).
% depth of this filter has to be the same as the depth of the input

% Common practice of filter size is 3x3 or at most 5x5.
% The dimensions of this filter is high x width x 3 for RGB.
% Filter is sliding, or convolving, around the input image.
% It is multiplying the values in the filter with the original pixel values
% of the image (aka computing element wise multiplications).


% The more filters, the greater the depth of the activation map,
% and the more information we have about the input volume.

% A common setting of the hyperparameters is F=3
filterSize = 3; % 3, 5, 11

layers = [
    % input image size is 128x128 and it is grayscale (1)
    imageInputLayer([128 128 1])
    
    % filterSize is pixel x pixel for high and width.
    % filterSize (F), number of filters (neurons),Name,Value
    % ex. 11,96,'Stride',4,'Padding',1
    convolution2dLayer(3, 64 ,'Padding','same')
    % 32 = 72
    % Batch normalization layers normalize the activations
    % and gradients propagating through a neural network
    % making network training an easier optimization problem.
    batchNormalizationLayer
    reluLayer
    
    % In practice, stride is 1, 2
    % Create a max pooling layer with nonoverlapping pooling regions
    % The most common form is a pooling layer with filters of size 2x2
    % applied with a stride of 2 downsamples
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3 ,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % POOL layer will perform a downsampling operation along the spatial dim.
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same');
    batchNormalizationLayer
    reluLayer
    
    % The OutputSize parameter in the last fully connected layer
    % is equal to the number of classes in the target data.
    fullyConnectedLayer(7)
    softmaxLayer;
    classificationLayer];



% Split train/test dataset
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

% Set option for training
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');

% trainNetwork, by default, shuffles the data at the beginning of training.
% trainNetwork function computes a dot product of the weights
% and the input, and then adds a bias term.
net = trainNetwork(imdsTrain,layers,options);


% Task - Plot the different filters of the first convolutional layer
% as you have done in part 1 and comment the results.
disp(net.Layers);
% Get the network weights for the convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights.
% There are xx individual sets of weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')


YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = (sum(YPred == YValidation)/numel(YValidation))*100;
fprintf('Accuracy %.2f percent', accuracy);