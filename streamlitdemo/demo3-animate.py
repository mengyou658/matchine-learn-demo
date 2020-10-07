import time

import streamlit as st
import pandas as pd
import numpy as np

progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(np.random.rand(10, 2))

for i in range(10):
  progress_bar.progress(i)
  new_rows = np.random.randn(10, 2)

  status_text.text("The lastest random number is : %s" % new_rows[-1, 1])
  chart.add_rows(new_rows)

  time.sleep(0.1)

status_text.text("done")
st.balloons()
# Draw a title and some text to the app:
'''
# This is the document title

1. This is some _markdown_.
'''

df = pd.DataFrame({'col1': [1,2,3]})
df  # <-- Draw the dataframe

x = 10
'x', x  # <-- Draw the string 'x' and then the value of x

with st.echo():
  st.write('This code will be printed')


st.help(pd.DataFrame)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
# X, Y value
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# plt.show()
st.pyplot()
