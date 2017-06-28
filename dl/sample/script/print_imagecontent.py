import sys

imgfile = sys.argv[1]

content = open(imgfile).read()
print content[:4]
