%% Generate IMDB - image database structure

% --------------------------------------------------------------------
% function imdb = getImdb(dataDir)
% --------------------------------------------------------------------
% Initialize the imdb structure (image database).
% Note the fields are arbitrary: only your getBatch needs to understand it.
% The field imdb.set is used to distinguish between the training and
% validation sets, and is only used in the above call to cnn_train.

% The sets, and number of samples per label in each set
%sets = {'train', 'val'} ;
%numSamples = [1500, 150] ;

dataDir = 'uncompressed_indexed_pictures';

% Preallocate memory
totalImages = 185; 
allFaces = zeros(224, 224, 3, totalImages, 'single') ; % VGG takes 224*224*3
%labels = zeros(totalImages, 1) ;
%set = ones(totalImages, 1) ;

% Read all samples
%sample = 1 ;
%for s = 1:2  % Iterate sets
%  for label = 1:3  % Iterate labels
    for i = 1:totalImages  % Iterate samples
      % Read image
      %im = imread(sprintf('%s/%s/%i/%04i.png', dataDir, sets{s}, label, i)) ;
      if i < 110 % 1 - 110 are png files
          im = imread(sprintf('%s/%i.png', dataDir, i));
      else
          im = imread(sprintf('%s/%i.jpg', dataDir, i));
      end
      
      % Resize it and store it, along with label and train/val set information
      im = imresize(im, [224, 224]);
      %im_ = single(im);
      allFaces(:,:,:,i) = single(im);
      %labels(sample) = label ;
      %set(sample) = s ;
      %sample = sample + 1 ;
    end
%  end
%end

% subtract averages
% since we are comparing two images, all of the images will be covered in
% training

imageMean = mean(allFaces(:,:,:,:),4);

% loop over ALL images to subtract Mean

for i = 1:size(allFaces,4)
    allFaces(:,:,:,i) = allFaces(:,:,:,i) - imageMean;
end
    
save('allFaces.mat','allFaces');

% Show some random example images
%figure(2) ;
%montage(images(:,:,:,randperm(totalSamples, 100))) ;
%title('Example images') ;

% Remove mean over whole dataset
%images = bsxfun(@minus, images, mean(images, 4)) ;

% Store results in the imdb struct
%imdb.images = images ;
%imdb.labels = labels ;
%imdb.set = set ;