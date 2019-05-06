clear all
close all
clc

%% read training dataset (we only choose 10 classes in the food-101 dataset)
digitDatasetPath = './food-101/10 classes';
digitData = imageDatastore(digitDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%% fix the image size from the original dataset
%  (we change the orignal images into 128*128 size grayimages.)

% nImg = length(digitData.Labels);
% for i = 1:nImg
%     a = digitData.Files(i);
%     str = char(a);
%     Img = imread(str);
%     if size(Img,3) == 3
%         ImgGray = rgb2gray(Img);
%     else
%         ImgGray = Img;
%     end
%     ImgResize = imresize(ImgGray,[128,128]);
%     imwrite(ImgResize,str);
% end

%% Display some of the images in the training dataset
%  (there are 9900 images for traing (990 images for each label)
figure;
perm = randperm(9900,20);
for i = 1:20
    subplot(4,5,i);
    imshow(digitData.Files{perm(i)});
end

%% output the labels and image size in the traing dataset 
%  (to make sure we have 10 labels' images and fixed the images' size)
labelCount = countEachLabel(digitData)
img = readimage(digitData,1);
size(img)

%% Specify Training and Validation Sets
%  (we use 750 images for training and 240 images for validation for each
%  labels)
trainNumFiles = 750;
[trainDigitData,valDigitData] = splitEachLabel(digitData,trainNumFiles,'randomize');

%% Define CNN Architecture
%  We defined three kinds of CNN architectures with different complexity,
%  and compared their speed and performance.And we also added a dropout 
%  layer before the fully connected layer to avoid overfitting.)

%  1. CNN with 5 convolution layers and 1 fully connected layer, and the number of 
%     filters in each convolution layer is 16,16,16,32,32. This one is relatively
%     simple and has comparable accuracy (about 45%).

layers = [
    imageInputLayer([128 128 1])
    
    convolution2dLayer(5,16,'Padding',2)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];

%  2. CNN with 3 convolution layers and 1 fully connected layer, and the number of 
%     filters in each convolution layer is 16,32,32. This one is the simplest one
%     because it only has three convolution layers. But the accuracy is much lower
%     than others (about 35%).

% layers = [
%     imageInputLayer([128 128 1])
%     
%     convolution2dLayer(5,16,'Padding',2)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(3,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%        
%     dropoutLayer
%     
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer
%     ];

% 3. CNN with 5 convolution layers and 1 fully connected layer, and the number of 
%     filters in each convolution layer is 32,32,64,128,128. This one is the most
%     complicated one because of much more filters in each convolution layer. 
%     But by simply increasing the number of filters, we observed that the final
%     accuracy has litter improvement (about 45%~50%). Considering the
%     experiment results in the paper, which was about 56% via a much more
%     complicated CNN architecture, we guess that the complexity of the origianl 
%     dataset might make the classification difficult. So the final accuracy 
%     would be limited (lower than 50%) by just using the simple CNN
%     architecture like our experiments.
%
% layers = [
%     imageInputLayer([128 128 1])
%     
%     convolution2dLayer(5,32,'Padding',2)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(3,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,64,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,64,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,128,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
%     dropoutLayer
%     
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer
%     ];


%% Specify Training Options
%  (We did several experiments to find the proper value of the training option
%   parameters. First, we use adam method to update the weights, which have 
%   better performance than the sgd method. Then we use 0.0001 learning rate 
%   to get a proper converge speed. And last we choose 10 epochs to get a more 
%   accurate result and reduce the influence of overfitting for the training
%   dataset.)

options =  trainingOptions('adam',...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',10, ...
    'MiniBatchSize',64,...
    'ValidationData',valDigitData,...
    'ValidationFrequency',30,...
    'Verbose',true,...
    'Plots','training-progress','ExecutionEnvironment','gpu');


%% Train the network
net = trainNetwork(trainDigitData,layers,options);

%% Classify validation images and compute accuracy using the final weights
predictedLabels = classify(net,valDigitData);
valLabels = valDigitData.Labels;
accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

%% After traing, using the final weights to classify the testing images
%  (10 testing images for each label)
testDatasetPath = './food-101/test images';
testData = imageDatastore(testDatasetPath,...
    'IncludeSubfolders',true,'LabelSource','foldernames');
estimatedLabels = classify(net,testData);
testLabels = testData.Labels;

testaccuracy = sum(estimatedLabels == testLabels)/numel(testLabels)

% randomly show the classifications of testing images
figure;
perm = randperm(100,20);
for i = 1:20
    subplot(4,5,i);
    imshow(testData.Files{perm(i)});
    imgTitle = string(testLabels(perm(i))) + '(' + ...
        string(estimatedLabels(perm(i))) + ')';
    title(imgTitle); % title format: real Labels(estimated Labels via our network)
end