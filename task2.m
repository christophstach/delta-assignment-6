net = googlenet;
dsFolder = './cars_motorcycles_pedestrians';
datastore = imageDatastore(dsFolder, 'IncludeSubfolders',true,...
        'LabelSource','FolderNames');
datastore.ReadFcn = @transformImageTask2;
    
labels = unique(datastore.Labels);
numClasses = length(labels);

[trainDataset, validationDataset] = splitEachLabel(datastore, 0.7, 0.3, 'randomized');





lgraph = layerGraph(net);
lgraph = replaceLayer(lgraph, 'loss3-classifier', [
    fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
]);
lgraph = replaceLayer(lgraph, 'prob', [
    softmaxLayer('Name', 'softmax')
]);
lgraph = replaceLayer(lgraph, 'output', [
    classificationLayer('Name', 'classoutput')
]);


figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
plot(lgraph);


opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, ...
'ValidationData', validationDataset, ...
'Plots', 'training-progress', ...
'MiniBatchSize', 64, ...
'ValidationPatience', 3);


tic
net = trainNetwork(trainDataset, lgraph, opts);
toc

function image = transformImageTask2(filename)
    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 
    image = imread(filename);
    image = image(:,:,min(1:3, end)); 
    image = imresize(image, [224 224]);
end