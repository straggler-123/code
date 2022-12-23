function [Idx, Center] = K_means(X, xstart,k)
% K-means聚类
% Idx是数据点属于哪个类的标记，Center是每个类的中心位置
% X是全部二维数据点，xstart是类的初始中心位置

len = length(X);        %X中的数据点个数
Idx = zeros(len, 1);    %每个数据点的Id，即属于哪个类

for nn=1:k
    C(nn,:) = xstart(nn,:);   
end

for i_for = 1:1000
    %为避免循环运行时间过长，通常设置一个循环次数
    %或相邻两次聚类中心位置调整幅度小于某阈值则停止
    
    %更新数据点属于哪个类
    for i = 1:len
        x_temp = X(i,:);    %提取出单个数据点
        for mm = 1:k
            d(mm) = norm(x_temp - C(mm,:));    
        end
     
        [~, id] = min(d);   %离哪个类最近则属于那个类
        Idx(i) = id;
    end
    
    %更新类的中心位置
  
    for ll=1:k
        C(ll,:) = mean(X(Idx == ll,:));
    end
end
for hh=1:k
    Center(hh,:) = C(hh,:);
end
  %类的中心位置

end
