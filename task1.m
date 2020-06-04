load('SelfTrainedMNISTModel.mat');

dsFolder = './sylvia_mnist';
testData = imageDatastore(dsFolder, 'IncludeSubfolders',true,...
        'LabelSource','FolderNames');
testData.ReadFcn = @transformImageTask1;
testTargets = testData.Labels;

testOutputs = net.classify(testData);
plotconfusion(testTargets, testOutputs);


function image = transformImageTask1(filename)
    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 
    image = imread(filename);

    image = rgb2gray(image);
    image = imresize(image, [28 28]);
    % imageSize = size(image);

    %if imageSize(1) < imageSize(2)
    %    image = imresize(image, [28 NaN]);   
    %else
    %    image = imresize(image, [NaN 28]);
    %end

    %cropper = centerCropWindow2d(size(image), [28, 28]);
    %image = imcrop(image, cropper);
    
    
    image = imcomplement(image);
    image = imbinarize(image, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.65);
    image = image .* 255;
    
end