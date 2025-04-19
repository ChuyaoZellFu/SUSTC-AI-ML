from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='The Round Table')

# 添加节点和边
dot.node('A', 'King Arthur')
dot.node('B', 'Sir Bedevere the Wise')
dot.node('L', 'Sir Lancelot the Brave')

dot.edges(['AB', 'AL'])
dot.edge('B', 'L', constraint='false')

# 输出DOT源码
print(dot.source)

# 保存和渲染图形，输出为PDF文件
dot.render('test-output/round-table.gv', view=True)