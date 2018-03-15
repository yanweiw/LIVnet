clear;clc;

% Load Pre-trained Model
net = load('vgg-face.mat') ;
layer = 34;
totalTrainNum = 5000;

% All our 4000 training images are stored in a 4-D matrix. 
% image4D(rw,col,ch,idx)--- row, column, channel, index
% B is a 4000-by-3 matrix

faces = load('allFaces.mat'); % 224*224*3*185
data = load('ordered_data_final.mat'); % 5000 * 3

train = data.ordered_data(1:totalTrainNum,:); % temporarily choose 1001-4000 tests to be training
label = train(:,3);

% Get feature matrix
    % These 2 feature matrix are m-1-n or 1-m-n
    % m: number of training sample(pairs)
    % n: number of features of the layer we choose to use
    FtMat1 = zeros(size(train,1), 4096); 
    FtMat2 = zeros(size(train,1), 4096);

for i = 1:size(train,1)
    
    fprintf('\nTraining data number %d\n', i);
    
    idx1 = train(i,1);  
    idx2 = train(i,2);
    
    img1 = faces.allFaces(:,:,:,idx1);
    img2 = faces.allFaces(:,:,:,idx2);
    
    % Apply VGG16
    res1 = vl_simplenn(net, img1) ;
    res2 = vl_simplenn(net, img2) ;
    
    % get feature matrix dimension
    [a, b, c] = size(res1(layer+1).x);
    
    % straighten out into one vector
    f_vec1 = reshape(res1(layer+1).x, [1, a*b*c]);
    f_vec2 = reshape(res2(layer+1).x, [1, a*b*c]);
    
    FtMat1(i, :) = f_vec1(1,:);
    FtMat2(i, :) = f_vec2(1,:);
    
end


save(sprintf('L%dresults.mat',layer),'FtMat1', 'FtMat2', 'label');
    
    
    
    
    
   
    
