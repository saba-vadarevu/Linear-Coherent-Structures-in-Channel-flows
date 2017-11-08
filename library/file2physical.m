function [] = file2physical(uHandle, sHandle, fName,options)
    load(fName)
    nx = 64; nz = 64; x0 = 0; x1 = 2*pi/aArr(1);
    z0 = -pi/bArr(1); z1 = pi/bArr(1);
    if nargin == 4
        nx = getDict(options,'nx',nx);
        nz = getDict(options,'nz',nz);
        x0 = getDict(options,'x0',x0);
        x1 = getDict(options,'x1',x1);
        z0 = getDict(options,'z0',z0);
        z1 = getDict(options,'z1',z1);
        
    end
    
    xArr = linspace(x0,x1,nx);
    zArr = linspace(z0,z1,nz);
    [yArr, ~] = chebdif(N,1);
    
    