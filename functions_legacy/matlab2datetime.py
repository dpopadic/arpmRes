import datetime as dt

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=int(matlab_datenum) % 1) - dt.timedelta(days = 366)
    return day + dayfrac

if __name__ == '__main__':
    print(matlab2datetime(731550))
