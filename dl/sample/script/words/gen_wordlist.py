# -*- coding: utf8 -*-
import sys

id = 0
with open(sys.argv[1]) as fp:
	for line in fp:
		line = line.strip()
		for i in range(0,len(line)-3,3):
			print "%s\t%s" % (id, line[i:i+3])
			id += 1
