function [labels, images] = readTrainData(maxCount)





%参考 http://yann.lecun.com/exdb/mnist/

%读取图片labels
fid=fopen('data/train-labels.idx1-ubyte');
magNum_l=fread(fid,1,'int','ieee-be')
%获取labels数量
itemCount_l=fread(fid,1,'int','ieee-be')
%获取labels
labels=fread(fid,min([maxCount itemCount_l]),'uint8','ieee-be');

%读取图片
fid=fopen('data/train-images.idx3-ubyte');
magNum_i=fread(fid,1,'int','ieee-be')
%获取图片数量
itemCount_i=fread(fid,1,'int','ieee-be')
%得到图片的高
rows_i=fread(fid,1,'int','ieee-be')
%得到图片的宽
columns_i=fread(fid,1,'int','ieee-be')
%读取图片
images= fread(fid,[rows_i*columns_i min([maxCount itemCount_i])]);

%img = reshape(image(:,2),rows_i,columns_i)';
%imshow(img)

end;




