```markdown
# 项目环境部署指南

## 1. 下载并安装 Anaconda

### 1.1 下载 Anaconda
- 可以选择用矩池云的服务器，其内置已安装 Anaconda。
- 也可以点击 [Anaconda 官网下载链接](https://www.anaconda.com/download/success)。
- 找到对应的 Linux 版本，这里选择的是 Anaconda3-2023 版本。下载到服务器，可以执行以下指令：

  ```sh
  apt update -y && wget -c --user-agent="Mozilla" 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2023.09-0-Linux-x86_64.sh'
  ```

### 1.2 安装 Anaconda
- 执行以下命令安装 Anaconda：

  ```sh
  bash Anaconda3-2023.09-0-Linux-x86_64.sh
  ```

- 安装过程中按回车键，直到出现 `yes or no` 选项，选择 `yes`。例如：

  ```
  Do you accept the license terms? [yes|no] [no] >>> yes
  ```

- 继续按回车键，直到安装完成。
- 退出并重新进入命令行，若命令行前出现 `(base)`，则说明安装成功。

## 2. 环境部署

### 2.1 创建并激活 Conda 环境
- 执行以下命令创建一个新的 Conda 环境（这里以 `xxxxx` 为环境名称）：

  ```sh
  conda create -n xxxxx python=3.9.13
  ```

- 若使用 `python=3.9.19`，需要修改 `setup.py` 文件中的 Python 版本。

- 激活 Conda 环境：

  ```sh
  conda activate xxxxx
  ```

### 2.2 克隆项目代码
- 将项目代码克隆到本地，执行以下命令：

  ```sh
  git clone https://gitee.com/DroneUE/nuaauav.git
  ```

- 按提示输入 Gitee 的用户名和密码。

### 2.3 安装项目依赖
- 进入项目目录：

  ```sh
  cd nuaauav
  ```

- 安装项目依赖：

  ```sh
  pip install -r requirements.txt
  ```

- 安装项目：

  ```sh
  pip install -e .
  ```

### 2.4 运行示例代码
- 进入 `drones` 目录并运行示例代码：

  ```sh
  cd drones && python demo.py
  ```

- 若出现未安装 `torch` 的报错，执行以下命令安装 `torch` 后再运行示例代码：

  ```sh
  pip install torch
  python demo.py
  ```

至此，项目环境部署完成。
```

如果有其他需要补充或修改的部分，请随时告知。