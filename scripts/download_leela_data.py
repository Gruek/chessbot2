#https://s3.amazonaws.com/lczero/training/games8620000.tar.gz

import urllib.request as request
import os

# 11 to 700
for i in range(896, 905):
    url = 'https://s3.amazonaws.com/lczero/training/games' + str(i) + '0000.tar.gz'
    # print(url)
    fname = 'games' + str(i) + '0000.tar.gz'
    if os.path.isfile(fname):
        pass
    else:
        print(url)
        request.urlretrieve(url, fname)
