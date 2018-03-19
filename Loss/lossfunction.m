save_path = '/Users/kaili/PycharmProjects/RWGT/data/dtd/';
s = 112;
imgDataPath = '/Users/kaili/PycharmProjects/RWGT/dtd/images/';

base = fft(eye(s)) / sqrt(s);
two_dim_base = kron(base,base);

imgDataDir  = dir(imgDataPath);             % ??????
for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... % ?????????????
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                % ???????????
           continue;
    end
    imgDir = dir([imgDataPath imgDataDir(i).name '/*.jpg']);
    saveDir = [save_path imgDataDir(i).name '.csv'];
    save_list = [];
    for j =1:length(imgDir)                 % ??????
        img = imread([imgDataPath imgDataDir(i).name '/' imgDir(j).name]);
        img = imresize(img,[s,s]);
        img = rgb2gray(img);
        temp = fft2(img);
        temp = reshape(temp,1,[]);
        temp = transpose(temp);
        para = two_dim_base\temp;
        para
        save_list = [save_list;para];
    end
    csvwrite(saveDir,save_list);
end