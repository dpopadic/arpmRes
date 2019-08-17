# http data download setup (default certificates)..
# 1. python -m pip install certifi
# 2. in terminal: /Applications/Python\ 3.7/Install\ Certificates.command

import urllib.request

# download specifications
user = 'dario.popadic@investecmail.com'
url_1 = 'https://www.arpm.co/lab/code/server/python/user/'
url_2 = '/lab/tree/databases/global-databases/'
file = 'db_BondAttribution.mat'
url = url_1 + user + url_2 + file
url_store = 'databases/global-databases/'

# store the file in the global-databases folder
urllib.request.urlretrieve(url, url_store + file)







