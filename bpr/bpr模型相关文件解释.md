### bpr模型相关文件解释
1. code文件夹是代码
2. data文件夹存储数据
3. notebook存储论文笔记

#### code文件夹解析

1. util_helper.py: 辅助工具包
2. train.py: 训练模型


#### 文件代码执行
	cd code
	# 训练在code文件夹下执行
	nohup python train.py > ./log/bpr_train.out 2>&1&
	
### Attention
1. 本次只是测试model,并未保存模型
	