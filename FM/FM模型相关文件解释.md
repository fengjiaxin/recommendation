### FM模型相关文件解释
1. code文件夹是代码
2. data文件夹存储数据
3. model文件夹存储模型文件
4. notebook存储论文笔记

#### code文件夹解析

1. hparams.py: 超参数设置
2. load_data.py: 载入数据工具包
3. model.py: fm模型类
4. util.py: 辅助文件工具包
5. train.py: 训练文件

#### 文件代码执行
	cd code
	# 训练在code文件夹下执行
	nohup python train.py > ./logs/fm_train.out 2>&1&
	
	# 查看tensorboard
	# 进入到model文件
	cd ..
	# 9765是端口号，可以自己设置
	tensorboard --port=9876 --logdir=./model/
	# 之后启动网页
	http://39.104.21.48:9876
	
	
### 后续代码提升
1. 可以在训练的时候同时测试验证数据集合
2. 代码读取数据的方式可以进行改进
	