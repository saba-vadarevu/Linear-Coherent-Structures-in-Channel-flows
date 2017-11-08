% Allows passing arrays by reference
classdef arrayHandle < handle
    properties
        array
    end
    methods
        function add(obj,someArr)
            obj.array = obj.array + someArr; 
        end
    end
end