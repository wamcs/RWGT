path1 = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /radish_original.png/o/coefficient.csv';
path2 = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /stone_original.png/o/coefficient.csv';
path3 = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /forest_original.png/o/coefficient.csv';
csv_path = '/Users/kaili/PycharmProjects/RWGT/result/texture_origin /compare.csv';
a = csvread(path1);
b = csvread(path2);
c = csvread(path3);
a = a(:,1:9);
b = b(:,1:9);
c = c(:,1:9);
ma1 = mean(abs(transpose(a)),1);
mb1 = mean(abs(transpose(b)),1);
mc1 = mean(abs(transpose(c)),1);
maxa = max(abs(transpose(a)));
maxb = max(abs(transpose(b)));
maxc = max(abs(transpose(c)));
va = var(abs(transpose(a)));
vb = var(abs(transpose(b)));
vc = var(abs(transpose(c)));
whole = cat(2,a,b);
whole = cat(2,whole,c);
whole = transpose(whole);

m = mean(whole,1);
temp_list = whole - repmat(m,27,1);
dis = abs(max(temp_list));
normal_list = temp_list./repmat(dis,27,1);
    
m = mean(abs(normal_list),1);
v = var(abs(normal_list),1);
a = angle(normal_list);
av = var(a);

s = [ma1;mb1;mc1;maxa;maxb;maxc;va;vb;vc;m;v;av];
csvwrite(csv_path,transpose(s));