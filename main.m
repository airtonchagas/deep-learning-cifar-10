close all; clear; clc;
load('net.mat');
load('train.mat');
%load('dados.mat');
%% carregando train
TTrain = train{1:50000,2};

for c = 1:50000
    XTrain(:,:,:,c) = imread(strcat('Train/Train/Img', int2str(c), '.png')); 

end

%% Define arquitetura CNN
conv1 = convolution2dLayer(5,32,'Padding',2,...
                     'BiasLearnRateFactor',2);
conv1.Weights = gpuArray(single(randn([5 5 3 32])*0.0001));
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
fc1.Weights = gpuArray(single(randn([64 576])*0.1));
fc2 = fullyConnectedLayer(10,'BiasLearnRateFactor',2);
fc2.Weights = gpuArray(single(randn([10 64])*0.1));

layers = [ ...
    imageInputLayer([32 32 3]);
    conv1;
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    fc2;
    softmaxLayer()
    classificationLayer()];

% opcoes
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 16, ...
    'Verbose', true);


%% Treinando CNN
[net, info] = trainNetwork(XTrain, TTrain, layers, opts);
save('net.mat', 'net')


%% Carregando test

TTest = train{40001:50000,2};

tic;
for c = 1:10000
    XTest(:,:,:,c) = imread(strcat('Test/Test/Img', int2str(c), '.png')); 

end

%% Classificando

YTest = classify(net, XTest);


%% calcula acuracia
% YTest = YTest(40001:50000);
% accuracy = sum(YTest == TTest)/numel(TTest)