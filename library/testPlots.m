function [] = testPlots(fName, level, tit, saveName)
load(fName)
szZ = size(zArr); szZ = szZ(2);
szU = size(u); szU = szU(2);
if szZ>szU
    zArr = zArr(1:szU);
end
[X,Z,Y] = meshgrid(zArr,xArr,yArr);

figure
%title(tit)

%level = 10;
subplot(2,2,1);
isosurface(X,Z,Y, u, level);
isosurface(X,Z,Y, u, -level); view(90,0);% colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
%title(fName)
box on
h = findobj(gcf,'type','axes');


subplot(2,2,2);
isosurface(X,Z,Y, u, level);
isosurface(X,Z,Y, u, -level); view(11,40);% colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
%title(fName)
box on


level = 1000;
subplot(2,2,3);
isosurface(X,Z,Y, u, level);
isosurface(X,Z,Y, u, -level); view(90,0);% colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
%title(fName)
box on

subplot(2,2,4);
isosurface(X,Z,Y, u, level);
isosurface(X,Z,Y, u, -level); view(11,40);% colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
%title(fName)
box on

supertitle(tit)
saveas(gcf, saveName, 'epsc');
end