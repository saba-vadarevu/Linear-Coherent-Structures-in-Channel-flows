function [] = plotVel(fName, level)
load(fName)
szZ = size(zArr); szZ = szZ(2);
szU = size(u); szU = szU(2);
if szZ>szU
    zArr = zArr(1:szU);
end
[X,Z,Y] = meshgrid(zArr,xArr,yArr);
figure
if false 
    [faces,verts,colors] = isosurface(X,Z,Y, u, level,u);
    patch('Vertices', verts, 'Faces', faces, 'FaceVertexCData',colors, ...
        'FaceColor', 'interp', 'EdgeColor', 'interp');
    [faces,verts,colors] = isosurface(X,Z,Y, u, -level,u);
    patch('Vertices', verts, 'Faces', faces, 'FaceVertexCData',colors, ...
        'FaceColor', 'interp', 'EdgeColor', 'interp');
elseif true
    p = patch(isosurface(X,Z,Y, u, level));
    isonormals(X,Z,Y, u, p);
    p.FaceColor = 'red';
    p.EdgeColor = 'none';
    p = patch(isosurface(X,Z,Y, u, -level));
    isonormals(X,Z,Y, u, p);
    p.FaceColor = 'green';
    p.EdgeColor = 'none';
    %camlight
    %lighting gouraud
    h = light;
    h.Position = [50, 20, 2];
else
    isosurface(X,Z,Y,u,level,u);
    isosurface(X,Z,Y,-u,level,u);
end
view(65,25); 
colorbar();
xlabel("z"); ylabel("x"); zlabel("y")
box on
%axis([-20, 20, 0, 160, -1., 1.])
end
