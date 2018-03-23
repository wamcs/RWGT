function deal_picture(img,path,two_dim_base,statistics,rgb)
%DEAL_PICTURE Summary of this function goes here
%   Detailed explanation goes here
if ~exist(path,'dir')
   mkdir(path)
end
save_path = [path '/'];
csv_path = [save_path 'coefficient.csv'];
imwrite(img,[save_path 'origin.jpg'])
if rgb == 1
    img = im2double(rgb2gray(img));
else
    img = im2double(img);
end
n = 100;
raw = -1;
save_list = [];
    for k = 1:9        
        if mod(k-1,3) == 0
            raw = raw + 1;        
        end
        raw*n+1
        mod(k-1,3)*n+1
        temp = imcrop(img,[raw*n+1 mod(k-1,3)*n+1 n-1 n-1]); 
        size(temp)   
        imwrite(temp,[save_path 'crop_' int2str(k) '.jpg'])
        t = fft2(temp);
        t = fftshift(t);
        t = log(abs(t)+1);
        imwrite(t,[save_path 'freq_' int2str(k) '.jpg'])
        if statistics == 1
            tempImg = reshape(temp,1,[]);
            size(tempImg)
            tempImg = transpose(tempImg);
            para = two_dim_base^-1*tempImg;
            save_list = [save_list;transpose(para)];
        end
    end
    if statistics == 1
        m = mean(save_list,1);
        temp_list = save_list - repmat(m,9,1);
        dis = abs(max(temp_list));
        normal_list = temp_list./repmat(dis,9,1);
    
        m = mean(abs(normal_list),1);
        v = var(abs(normal_list),1);
        a = angle(normal_list);
        av = var(a);
    
        save_list = [save_list;m];
        save_list = [save_list;v];
        save_list = [save_list;av];
        csvwrite(csv_path,transpose(save_list));
    end
end

