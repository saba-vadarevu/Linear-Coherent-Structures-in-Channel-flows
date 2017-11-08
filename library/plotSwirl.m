function [] = plotSwirl(fName, level)
load(fName)
szZ = size(zArr); szZ = szZ(2);
szU = size(u); szU = szU(2);
if szZ>szU
    zArr = zArr(1:szU);
end
[X,Z,Y] = meshgrid(zArr,xArr,yArr);
figure
isosurface(X,Z,Y, swirl, level,vorz); view(110,25); colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
title(fName)
box on
%axis([-20, 20, 0, 150, -1, 1])
end