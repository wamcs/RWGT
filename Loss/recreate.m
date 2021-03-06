% 
% function recreate(file_path,save_path,two_dim_base)
% if ~exist(save_path,'dir')
%    mkdir(save_path)
% end
% m = csvread(file_path);
% m = m(:,1:9);
% [r,l] = size(m);
% % m1 = cat(1,m(1:2000,:),zeros(10000-2000,9));
% m1 = cat(1,zeros(2000,9),m(2001:r,:));
% temp1 = two_dim_base*m1;
% for i = 1:9
%     temp = temp1(:,i);
%     temp = reshape(temp,[100,100]);
%     imwrite(temp,[save_path '/recreate_' int2str(i) '.jpg']);
% end
% 
% end


% function recreate(r_path,g_path,b_path,save_path,n,two_dim_base)
% if ~exist(save_path,'dir')
%    mkdir(save_path)
% end
% m1 = csvread(r_path);
% m2 = csvread(g_path);
% m3 = csvread(b_path);
% m1 = m1(:,1:9);
% m2 = m2(:,1:9);
% m3 = m3(:,1:9);
% % m1 = cat(1,m(1:2000,:),zeros(10000-2000,9));
% % m1 = cat(1,zeros(2000,9),m1(2001:n*n,:));
% % m2 = cat(1,zeros(2000,9),m2(2001:n*n,:));
% % m3 = cat(1,zeros(2000,9),m3(2001:n*n,:));
% temp1 = two_dim_base*m1;
% temp2 = two_dim_base*m2;
% temp3 = two_dim_base*m3;
% for i = 1:9
%     r = temp1(:,i)/0.299;
%     g = temp2(:,i)/0.587;
%     b = temp3(:,i)/0.114;
%     r = reshape(r,[n,n]);
%     g = reshape(g,[n,n]);
%     b = reshape(b,[n,n]);
%     temp = zeros(n,n,3);
%     temp(:,:,1) =  r*255;
%     temp(:,:,2) =  g*255;
%     temp(:,:,3) =  b*255;
%     imshow(temp)
%     imwrite(r,[save_path '/recreate_' int2str(i) '.jpg']);
% end
% 
% end

 r_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /radish_original.png/r/coefficient.csv';
 g_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /radish_original.png/g/coefficient.csv';
 b_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /radish_original.png/b/coefficient.csv';
 save_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /radish_original.png/recreate/o';

% r_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /stone_original.png/r/coefficient.csv';
% g_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /stone_original.png/g/coefficient.csv';
% b_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /stone_original.png/b/coefficient.csv';
% save_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /stone_original.png/recreate/o';

%  r_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /forest_original.png/r/coefficient.csv';
%  g_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /forest_original.png/g/coefficient.csv';
%  b_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /forest_original.png/b/coefficient.csv';
%  save_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /forest_original.png/recreate/o';
n = 100;
base = fft(eye(n)) / sqrt(n);
two_dim_base = kron(base,base);


if ~exist(save_path,'dir')
   mkdir(save_path)
end
m1 = csvread(r_path);
m2 = csvread(g_path);
m3 = csvread(b_path);
m1 = m1(:,1:9);
m2 = m2(:,1:9);
m3 = m3(:,1:9);

m1 = cat(1,m1(1:200,:),zeros(10000-200,9));
m2 = cat(1,m2(1:200,:),zeros(10000-200,9));
m3 = cat(1,m3(1:200,:),zeros(10000-200,9));

%  m1 = cat(1,zeros(9000,9),m1(9001:n*n,:));
%    m2 = cat(1,zeros(9000,9),m2(9001:n*n,:));
%   m3 = cat(1,zeros(9000,9),m3(9001:n*n,:));

% m1(1000:9900,:) = 0;
%  m2(1000:9900,:) = 0;
%  m3(1000:9900,:) = 0;
% 
%  m1(1:100,:) = 0;
%  m2(1:100,:) = 0;
%   m3(1:100,:) = 0;
% % 
%  m1(9000:n*n,:) = 0;
%  m2(9000:n*n,:) = 0;
%   m3(9000:n*n,:) = 0;

% m1(10:3:n*n,:) = 0;
% m2(10:3:n*n,:) = 0;
% m3(10:3:n*n,:) = 0;


temp1 = two_dim_base*m1;
temp2 = two_dim_base*m2;
temp3 = two_dim_base*m3;
for i = 1:9
    r = temp1(:,i);
    g = temp2(:,i);
    b = temp3(:,i);
    r = reshape(r,[n,n]);
    g = reshape(g,[n,n]);
    b = reshape(b,[n,n]);
    temp = zeros(n,n,3);
    temp(:,:,1) =  r*255;
    temp(:,:,2) =  g*255;
    temp(:,:,3) =  b*255;
    temp = uint8(temp);
    imwrite(temp,[save_path '/recreate_' int2str(i) '.jpg']);
end
