from numpy import where


def SectorSelect(sectors, sect):
    # This function returns the indeces of the entries in "sectors" that
    # coincide with the sector specified by "sect", i.e. sectors[index] == sect
    #  INPUT
    # sectors  :[string array] list of sectors
    # sect     :[string array] choosen sector
    #  OP
    # index    :[column vector] indeces of companies belonging to sector "sect"

    ## Code
    a = sect==sectors
    index = where(a)[0]
    return index
