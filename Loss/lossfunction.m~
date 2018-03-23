
save_path = '/Users/kaili/PycharmProjects/RWGT/result/';
s = 300;
num = 3;
imgDataPath = '/Users/kaili/PycharmProjects/RWGT/data/';

base = fft(eye(s/3)) / sqrt(s/3);
two_dim_base = kron(base,base);

imgDataDir  = dir(imgDataPath);             
for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... 
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                
           continue;
    end
    imgDir = dir([imgDataPath imgDataDir(i).name '/*.png']);
    saveDir = [save_path imgDataDir(i).name];
    if ~exist(saveDir,'dir')
        mkdir(saveDir)
    end
    for j =1:length(imgDir) 
        savePicPath = [saveDir '/' imgDir(j).name];
        if ~exist(savePicPath,'dir')
            mkdir(savePicPath)
        end
        img = imread([imgDataPath imgDataDir(i).name '/' imgDir(j).name]);
        img = imresize(img,[s,s]);
%         red = zeros(s,s,3);
%         green = zeros(s,s,3);
%         blue = zeros(s,s,3);
        
        red = img(:,:,1);
        green = img(:,:,2);
        blue = img(:,:,3);
        
%         deal_picture(img,[savePicPath '/' 'o'],two_dim_base,1,1)
          deal_picture(red,[savePicPath '/' 'r'],two_dim_base,1,0)
          deal_picture(green,[savePicPath '/' 'g'],two_dim_base,1,0)
          deal_picture(blue,[savePicPath '/' 'b'],two_dim_base,1,0)
            
    end
    
end


