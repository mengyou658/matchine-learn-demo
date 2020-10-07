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
plt.plot(x,y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')

plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('x坐标')
plt.ylabel('y坐标')
# plt.show()

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)

# 使用plt.yticks设置y轴刻度以及名称：刻度为[-2, -1.8, -1, 1.22, 3]；对应刻度的名称为[‘really bad’,’bad’,’normal’,’good’, ‘really good’]. 使用plt.show显示图像.
plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
plt.show()
