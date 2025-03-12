# 安装 requirements.txt 中的依赖项
install:
	pip3 install -r requirements.txt

# 构建 Docker 镜像
docker-build:
	docker build -t jbtrade:latest .

# 以交互模式运行 Docker 容器
docker-run:
	docker run -it --rm -v $(PWD):/app jbtrade:latest /bin/bash

# 生成或更新 requirements.txt
pipfreeze:
	pip3 freeze > requirements.txt

# 使用 pipreqs 生成或更新 requirements.txt
pipreqs:
	pip3 install pipreqs
	pipreqs . --force

# 清空生成的 .pyc 文件和 __pycache__ 目录
clean:
	# find . -name "*.pyc" -exec rm -f {} +
	find . -name "__pycache__" -exec rm -rf {} +
