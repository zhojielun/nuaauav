大体教程

1.首先下载Anaconda，可以选择用矩池云的服务器，内置就有安装
- 点击官网下载链接：https://www.anaconda.com/download/success
- 找到对应的Linux版本，这里我选择的是Anaconda3-2023的版本，然后下载到服务器里，可以执行这条指令：apt update -y && wget -c --user-agent="Mozilla" 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2023.09-0-Linux-x86_64.sh'
- 安装Anaconda，执行该命令：bash Anaconda3-2023.09-0-Linux-x86_64.sh
- 一直按回车，直到出现 yes or no 选项，选yes，比如说：Do you accept the license terms? [yes|no] ，[no] >>> yes，之后继续回车
- 应该就可以了，然后就退出重新进入命令行，在命令行前面出现(base)就正常了
2.开始进行环境部署
- 首先执行 conda create -n xxxxx python=3.9.13(或者python=3.9.19)，用3.9.19的话就要修改一下setup.py里的python版本。
- 然后激活conda环境，输入 conda activate xxxxx
- 然后将项目Git下来，执行git clone https://gitee.com/DroneUE/nuaauav.git ,按照提示正常输入gitee的用户名和密码
- 然后执行cd nuaauav
- 然后执行pip install -r requirements.txt
- 然后执行pip install -e .
- 最后执行cd drones && python demo.py ,这里可能出现报错，提示未安装torch，那就执行pip install torch，然后再执行python demo.py即可