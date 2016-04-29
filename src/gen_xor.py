import sys


def main(of, nbit):
	if not (of and nbit and nbit <= 20):
		raise Exception("wtf, 1 to 20 bits plz")
	limit = 2**nbit
	with open(of, "w") as f:
		bits = [0] * nbit
		for i in range(limit):
			for j in range(nbit):
				bits[j] = 1 if (i & (1<<(nbit-j-1))) else 0
			output = sum(bits) & 1
			f.write(' '.join([str(b) for b in bits]))
			f.write(' ' + str(output) + '\n')



if __name__ == '__main__':
	if len(sys.argv) != 3:
		print("Usage: %s [output filename] [n bits]" % sys.argv[0])
	else:

		main(sys.argv[1], int(sys.argv[2]))