function [] = plotVel(fName, level)
load(fName)
szZ = size(zArr); szZ = szZ(2);
szU = size(u); szU = szU(2);
if szZ>szU
    zArr = zArr(1:szU);
end
[X,Z,Y] = meshgrid(zArr,xArr,yArr);
figure
isosurface(X,Z,Y, u, level);
isosurface(X,Z,Y, u, -level); view(55,50); colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
box on
axis([-20, 20, 0, 160, -1., 1.])
end