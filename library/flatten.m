function [colVec] = flatten(arr)
    colVec = reshape(arr, [length(arr), 1] );
end