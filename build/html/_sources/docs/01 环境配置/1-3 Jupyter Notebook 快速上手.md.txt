## 1-3 Jupyter Notebook 快速上手

### Jupyter是什么

Jupyter Notebook是一个开源的Web应用程序，允许开发者方便的创建和共享代码文档

可以实时的编写代码块运行代码块，查看结果并可视化数据

### Jupyter Notebook特点
- 支持markdown语法
- 支持LaTeX公式   
- 允许把代码写入到独立的cell中，然后单独执行，无需从头开始执行代码

### 教程结构

![png](../../notebook/1-3/assets/1.png)

### 1. Jupyter的安装与运行

#### 1.1 图形化界面操作

##### 1.1.1 默认的base环境
安装了Anaconda后，在base环境就默认就安装了Jupyter Notebook

打开Anaconda，在home界面直接点击launch运行

![png](../../notebook/1-3/assets/2.png)

##### 1.1.2 创建新的虚拟环境
打开Anaconda,在Environments界面，创建新的python虚拟环境

![png](../../notebook/1-3/assets/3.png)

然后回到home界面，点击install按钮安装Jupyter  Notebook

![png](../../notebook/1-3/assets/4.png)

安装完成后点击launch按钮运行


#### 1.2 命令行操作
##### 1.2.1 默认的base环境
打开Anaconda Prompt命令行： 因为base环境默认已经安装了Jupyter Notebook

![png](../../notebook/1-3/assets/5.png)

所有直接输入命令： jupyter notebook  运行

![png](../../notebook/1-3/assets/6.png)

##### 1.2.2 创建新的虚拟环境
打开Anaconda Prompt命令行：首先创建一个新的虚拟环境

conda create -n Gengzhi python=3.7

然后切换到Gengzhi环境：conda activate Gengzhi

然后进行安装：  conda install jupyter notebook

安装完成后输入命令运行：  jupyter notebook

