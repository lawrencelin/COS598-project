import sys
import matplotlib.pyplot as plt

def main(fname, NT):
	x = [i+1 for i in range(NT)]
	x.reverse()
	t = []
	for line in open(fname):
		if line.find('real') != 0: continue
		t.append(float(line.split()[-1]))
		if (len(t) == NT): break

	t_0 = t[0]
	for i in range(NT):
		t[i] /= t_0
	y = []
	for i in range(NT):
		y.append(t[0] / (t[i] * (i+1)))
	plt.barh(x, t, edgecolor='white')

	for _x,_t in zip(x, t):
		plt.text(_t+0.05, _x+0.9, '%.2f' % _t, ha='center', va= 'top')
		plt.text(-0.03, _x, '%d' % (NT-_x+1), ha='center', va= 'bottom')

	plt.xlabel("Normalized Runtime")
	plt.ylabel("Number of Threads", labelpad=20)
	plt.xlim(0, 1.1)
	plt.ylim(0, NT+2)
	plt.xticks([])
	plt.yticks([])
	plt.show()

	plt.barh(x, y, edgecolor='white')

	for _x, _y in zip(x, y):
		plt.text(_y+0.05, _x+0.9, '%.2f' % _y, ha='center', va= 'top')
		plt.text(-0.03, _x, '%d' % (NT-_x+1), ha='center', va= 'bottom')

	plt.xlabel("Normalized Per-thread Throughput")
	plt.ylabel("Number of Threads", labelpad=20)
	plt.xlim(0, 1.1)
	plt.ylim(0, NT+2)
	plt.xticks([])
	plt.yticks([])
	plt.show()

def main1(fname):
	x = [float(line) / 1000.0 for line in open(fname)]
	plt.hist(x, bins=20, normed=True, edgecolor='white')
	plt.xlabel("Response Time (s)")
	plt.ylabel("Normalized Frequency")
	plt.yticks([])
	plt.show()

def main2():
	x = [8,7,6,5,4,3,2,1]
	t = [480.294, 242.366, 162.877, 125.39, 102.53, 85.339, 75.445, 68.037]
	t_0 = t[0]
	t = [_t / t_0 for _t in t]
	y = [t[0] / (t[i] * (i+1)) for i in range(8)]

	plt.barh(x, t, edgecolor='white')

	for _x,_t in zip(x, t):
		plt.text(_t+0.05, _x+0.9, '%.2f' % _t, ha='center', va= 'top')
		plt.text(-0.03, _x, '%d' % (8-_x+1), ha='center', va= 'bottom')

	plt.xlabel("Normalized Runtime")
	plt.ylabel("Number of Machines", labelpad=20)
	plt.xlim(0, 1.1)
	plt.ylim(0, 8+2)
	plt.xticks([])
	plt.yticks([])
	plt.show()

	plt.barh(x, y, edgecolor='white')

	for _x, _y in zip(x, y):
		plt.text(_y+0.05, _x+0.9, '%.2f' % _y, ha='center', va= 'top')
		plt.text(-0.03, _x, '%d' % (8-_x+1), ha='center', va= 'bottom')

	plt.xlabel("Normalized Per-Machine Throughput")
	plt.ylabel("Number of Machines", labelpad=20)
	plt.xlim(0, 1.1)
	plt.ylim(0, 8+2)
	plt.xticks([])
	plt.yticks([])
	plt.show()



if __name__ == '__main__':
	if len(sys.argv) == 3: main(sys.argv[1], int(sys.argv[2]))
	elif len(sys.argv) == 2: main1(sys.argv[1])
	else: main2()
		# print("usage: %s [filename] [NT]" % sys.argv[0])