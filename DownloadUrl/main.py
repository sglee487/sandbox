# https://gist.github.com/moo-gii/707cf3b77723691be47d2496d1148b23

# load library
import urllib.request
import os

# image url to download
url = "http://cfile30.uf.tistory.com/image/99BA21335A118CC2050938"

# file path and file name to download
outpath = "C:/test/"
outfile = "test.png"

# Create when directory does not exist
if not os.path.isdir(outpath):
    os.makedirs(outpath)

# download
urllib.request.urlretrieve(url, outpath+outfile)
print("complete!")