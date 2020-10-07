# streamlit 可视化的工具
暂时不支持中文展示，py文件中有中文会报错
修改 D:\Anaconda3\envs\demo-machine\Lib\site-packages\streamlit\ScriptRunner.py 247行
with open(self._report.script_path, encoding='UTF-8') as f:
# 源码编译
1. 下载 https://unpkg.com/viz.js@1.8.0/viz.js  public/vendor/viz/viz-1.8.0.min.js
1. protoc --proto_path=proto  --python_out=lib proto/streamlit/proto/*.proto
1. mkdir -p frontend/src/autogen
1. cd frontend && echo "/* eslint-disable */" > src/autogen/proto.js && .\node_modules\.bin\pbjs.cmd ../proto/streamlit/proto/*.proto -t static-module --es6 > src/autogen/proto.js
1. cd frontend && yarn run --silent scss-to-json src/assets/css/variables.scss > src/autogen/scssVariables.ts
1. start 更改 set NODE_OPTIONS --max_old_space_size=8000 && craco start

