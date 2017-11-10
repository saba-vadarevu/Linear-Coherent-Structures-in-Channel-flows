function [] = plotSwirl(fName, level,x0,x1)
load(fName)
szZ = size(zArr); szZ = szZ(2);
szU = size(u); szU = szU(2);
if szZ>szU
    zArr = zArr(1:szU);
end
[X,Z,Y] = meshgrid(zArr,xArr,yArr);
figure
arr = swirl; colors=vorz;
swirlMax = max( arr(:) );
vorzMax = max(colors(:));
if level < 1
    level0 = level;
    level = level * swirlMax;
else
    level0 = level/swirlMax;
end
if false 
    [faces,verts,colors] = isosurface(X,Z,Y, arr, level,colors);
    patch('Vertices', verts, 'Faces', faces, 'FaceVertexCData',colors, ...
        'FaceColor', 'interp', 'EdgeColor', 'interp');
    [faces,verts,colors] = isosurface(X,Z,Y, arr, -level,colors);
    patch('Vertices', verts, 'Faces', faces, 'FaceVertexCData',colors, ...
        'FaceColor', 'interp', 'EdgeColor', 'interp');
elseif false
    p = patch(isosurface(X,Z,Y, arr, level, colors));
    isonormals(X,Z,Y, arr, p);
    %p.FaceColor = 'red';
    %p.EdgeColor = 'none';
    p = patch(isosurface(X,Z,Y, arr, -level, colors));
    isonormals(X,Z,Y, arr, p);
    %p.FaceColor = 'green';
    %p.EdgeColor = 'none';
    %camlight
    %lighting gouraud
    h = light;
    h.Position = [50, 20, 2];
else
    isosurface(X,Z,Y,arr,level,colors);
    isosurface(X,Z,Y,-arr,level,colors);
    h = light;
    h.Position = [50, 20, 2];
end
title(["t=",num2str(t)]);
view(65,25); 
caxis([-0.05*vorzMax, 0.05*vorzMax]);
colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
box on
if nargin > 2 
    axis([-2, 2, x0, x1, -1., -0.])
end
end
