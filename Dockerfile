# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制 requirements.txt 并安装依赖项
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制当前目录的所有文件到工作目录
COPY . .

# 设置容器启动时的默认命令
CMD ["python", "get_k_line_hist.py"]
