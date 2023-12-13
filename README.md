# AI-brain-MRI
人工智能医疗影像——Brain MRI训练及预测,主要用U-net                 

文件'train&pred.py'是代码，你需要运行这个，但是文件中有我修改过的地方和注释了部分代码，所以需要根据情况修改（'train&pred(haveing difficulty).py'纯手打4小时，但是模型没训练（不会...）)假如用第二个代码需要把数据集里里面的data.csv?删掉       

数据集代码在kaggle中搜索brain MRI第一个就是，可以直接参考下面链接：             
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data    

由于训练模型需要9.9小时左右，所以直接提供model：   
https://www.kaggle.com/code/tmamirulhaquebhuiyan/braintumorunet/output        

out文件夹里面是部分预测输出对比图片（失真0.55，所以预测效果并不是很好，因为模型只训练了1小时）



