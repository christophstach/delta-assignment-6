load('vMNISTModel.mat');


outputs = [];
targets = [];

for i = 0:9
    directory = './sylvia_mnist/' + string(i) + '/';
    files = dir(directory + '*.jpg');
    for o = 1:length(files)
        name = files(o).name;
        image = imread(directory + name);
        image = rgb2gray(image);
        imageSize = size(image);

        if imageSize(1) < imageSize(2)
            image = imresize(image, [28 NaN]);   
        else
            image = imresize(image, [NaN 28]);
        end

        cropper = centerCropWindow2d(size(image), [28, 28]);
        image = imcrop(image, cropper);
        % image = im2double(image);
        image = imcomplement(image);
        output = net.classify(image);
        
        outputs = [outputs output];
        targets = [targets categorical(i)];
    end
end


t = table(targets.', outputs.');

plotconfusion(targets, outputs);
% fig = uifigure;
% uitable(fig, 'Data', t, 'ColumnName', {'Targets', 'Outputs'});



