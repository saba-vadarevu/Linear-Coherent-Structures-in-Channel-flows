function returnVal = getDict(dictName, keyName, defaultVal)
    if isKey(dictName,keyName)
        returnVal = dictName(keyName);
    else
        returnVal = defaultVal;
    end
end