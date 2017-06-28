import sys

with open(sys.argv[1]) as fp:
	color = []
	for line in fp:
		tokens = line.strip('\r\n').split('\t')
		if len(tokens) == 0 or len(tokens[0].strip()) == 0:
			continue
		val = tokens[0]
		if val[0].isalpha():
			if len(color) != 0:
				print "\t".join(color)
			color = []
			color.append(val)
		elif val[0] == "#":
			val_fix = "#" + val[5:7]+val[3:5]+val[1:3]
			color.append(val_fix)
		elif val[0].isdigit():
			color.append(val)
		else:
			pass
		
