import matplotlib.pyplot as plt
import numpy as np

# 在我的 notebook 里，要设置下面两行才能显示中文
plt.rcParams['font.family'] = ['sans-serif']
# 如果是在 PyCharm 里，只要下面一行，上面的一行可以删除
plt.rcParams['font.sans-serif'] = ['SimHei']

x = np.linspace(-3,3,50)
y1 = 2*x + 1
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

# 使用plt.gca获取当前坐标轴信息. 使用.spines设置边框：右侧边框；使用.set_color设置边框颜色：默认白色； 使用.spines设置边框：上边框；使用.set_color设置边框颜色：默认白色；
ax = plt.gca()
ax.spines['right'].set_color('red')
ax.spines['top'].set_color('blue')
# 使用.xaxis.set_ticks_position设置x坐标刻度数字或名称的位置：bottom.（所有位置：top，bottom，both，default，none）
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# plt.legend(loc='upper right')
# 其中’loc’参数有多种，’best’表示自动分配最佳位置，其余的如下：
#  'best' : 0,
#  'upper right'  : 1,
#  'upper left'   : 2,
#  'lower left'   : 3,
#  'lower right'  : 4,
#  'right'        : 5,
#  'center left'  : 6,
#  'center right' : 7,
#  'lower center' : 8,
#  'upper center' : 9,
#  'center'       : 10,
plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best')
plt.show()
