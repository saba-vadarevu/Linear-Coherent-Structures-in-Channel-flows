function vec1 = compress(vec)

vec1 = [];
n1 = size(vec);

for i = 1:n1
    if(vec(i) ~= 0) 
        vec1 = [vec1 vec(i)];
    end
end

vec1 = vec1';

end