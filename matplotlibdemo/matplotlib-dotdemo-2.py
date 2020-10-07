import matplotlib.pyplot as plt
import numpy as np

# 在我的 notebook 里，要设置下面两行才能显示中文
plt.rcParams['font.family'] = ['sans-serif']
# 如果是在 PyCharm 里，只要下面一行，上面的一行可以删除
plt.rcParams['font.sans-serif'] = ['SimHei']

x = np.arange(-1.0, 2.0, 0.1)
y1 = x * np.cos(np.pi * x)
y2 = x**2
plt.figure()
l1, = plt.plot(x, y2, label='liner line')
l2, = plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='square line')
# 画垂直虚线
x0 = 1
y0 = 2*x0 + 1
l3, = plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2.5)
plt.scatter([x0, ], [y0, ], s=50, color='b')
# 添加注释
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
# 添加注释 text
plt.text(-1.2, 1, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
         fontdict={'size': 16, 'color': 'r'})
plt.xlim((-1, 2))
plt.ylim((-2, 3))
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3],['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])

plt.show()
