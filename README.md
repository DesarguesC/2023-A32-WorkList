# 2023-A32-WorkList





## district-free guidance scale based few-shot time serie fitting method
English version

visit our full program at https://github.com/DesarguesC/2023-A32-WorkList

**Introduction**

![model](/data/assets/model.png)
Using LSTM and convolutional neural network, as well as renet, we solve the data correcting problem with finally $R^2\ge0.98$.

We propose an improvement of the using of the ResNet with an extra scale called *district-free guidance scale*, which can be simply denoted as *dfg-scale*. With this dfg-scale, our model gain the migration ability from district to district.

**Attention**: the *district-free guidance scale* improve the ability of the model in migration, generalization but not model's accurancy. Though the accurancy($R^2$) may actually enhanced by led in dfg-scale. 

On the other hand, our model is capable of self-study ability（automatic fine tune） on a few-shot dataset based on pretrained weights. Namely, show some data (few-shot dataset, with standard station data) to our pretrained-model, it can find a scale which makes $R^2$ criterion reached the most value on this few-shot term. After this searching step, the model find the scale which fit the few-shot term the most and then run on total dataset to amend the data.

### environment set up and dependencies install
To set up, we need $torch\ge1.11.0$. If pytorch has been installed under the base environment in anaconda, you can directly install our dependencies(packages) by running
```bat
pip install -r requirements.txt
```
in terminal.

Otherwise, you should install our environment from scratch by running
```bat
conda env create -f environment.yaml
```
in order to use our project.

### Run our program

As you clone the program, turn to folder 'test' and activate your virtual environment(if you have launched a new virtual environment for our program)
```bat
rectification
cd test
```
running our main code
```bat
python test.py
```

By default, this use district-free guidance scale = 1.0 on the default dataset path './data/A32.xlsx'. All supported arguments are listed below (type python test.py --help).
```bat
usage: test.py [-h] [--mini_station_num MINI_STATION_NUM] [--data_path DATA_PATH] [--pt_path PT_PATH] [--dfg_scale DFG_SCALE]
               [--norm {standard,max-min}] [--few_shot_mode FEW_SHOT_MODE] [--few_shot_length FEW_SHOT_LENGTH]
               [--few_shot_path FEW_SHOT_PATH] [--save_find_fig SAVE_FIND_FIG] [--base_scope BASE_SCOPE]

optional arguments:
  -h, --help            show this help message and exit
  --mini_station_num MINI_STATION_NUM
                        how many tiny stations are on the dataset
  --data_path DATA_PATH
                        the path of your dataset
  --pt_path PT_PATH     the path of your model weights
  --dfg_scale DFG_SCALE
                        district-free guidance scale as we dedcribed
  --norm {standard,max-min}
                        how to normalize the input data
  --few_shot_mode FEW_SHOT_MODE
                        to use zero shot mode
  --few_shot_length FEW_SHOT_LENGTH
                        the length of dataset and must be greater than 5
  --few_shot_path FEW_SHOT_PATH
                        the path that the few-shot sequence data is placed
  --save_find_fig SAVE_FIND_FIG
                        whether to save figure when searching the best scale
  --base_scope BASE_SCOPE
                        use the scope when find the best dfg-scale
```

NOTE: When you are running with a few-shot dataset input, remember to set --few_shot_mode=True, or the model fails to read your few-shot dataset.





## 基于免地域指引比例的小时域数据适应方法
中文版

在https://github.com/DesarguesC/2023-A32-WorkList上访问我们完整的项目

### 简介

使用LSTM记忆网络和卷积神经网络以及残差连接网络，我们解决数据矫正任务，所得最终的校验结果为$R^2\ge0.98$

我们提出了针对使用残差连接网络时的一个优化，通过过引入一个“免地域指引比例”，简写为“dfg-scale”。通过dfg-scale，我们的模型获得了将学习获得的效果在地域间迁移的能力。通过使用这个dfg-scale，我们的模型获得了从一个地区到另一个地区的迁移能力。

**注意**：dfg-scale的使用是模型的迁移能力和泛化能力得到了提高，而不会影响模型的准确率。尽管这个准确率($R^2$)可能实际上确实会被前者影响。

另一方面，我们的预训练模型拥有小数据集（few-shot）上的的自学能力（自动微调）。即，通过向我们的预训练模型投喂一定的数据（小数据集，有标准站数据），它能够找到一个使$R^2$最大的dfg-scale值。在搜索之后，模型能够找到是的在这个小数据集上表现最好的dfg-scale值，然后使用这个dfg-scale来矫正整个数据集上的数据。



### 环境部署与依赖安装
先部署我们的环境，首先要求保证torch$\ge 1.11.0$，如果在anaconda的base环境中已经安装过pytorch，那么可以直接按照我们的包依赖，在终端运行
```bat
pip install -r requirements.txt
```
否则，需要先运行
```bat
conda env create -f environment.yaml
```
才能使用我们的项目

### 项目运行

当clone了我们的项目后，在终端中打开我们的项目文件夹，切换进入test文件夹并激活虚拟环境（如果有为我们的项目申请出一个虚拟环境的话
```bat
conda activate rectification
cd test
```
运行我们的主程序代码
```bat
python test.py
```
此时使用默认值dfg-scale=1.0以及默认数据集地址data/A32.xlsx. test.py中所有的支持的参数值如下表所示（输入python test.py --help）
```bat
usage: test.py [-h] [--mini_station_num MINI_STATION_NUM] [--data_path DATA_PATH] [--pt_path PT_PATH] [--dfg_scale DFG_SCALE]
               [--norm {standard,max-min}] [--few_shot_mode FEW_SHOT_MODE] [--few_shot_length FEW_SHOT_LENGTH]
               [--few_shot_path FEW_SHOT_PATH] [--save_find_fig SAVE_FIND_FIG] [--base_scope BASE_SCOPE]

optional arguments:
  -h, --help            show this help message and exit
  --mini_station_num MINI_STATION_NUM
                        how many tiny stations are on the dataset
  --data_path DATA_PATH
                        the path of your dataset
  --pt_path PT_PATH     the path of your model weights
  --dfg_scale DFG_SCALE
                        district-free guidance scale as we dedcribed
  --norm {standard,max-min}
                        how to normalize the input data
  --few_shot_mode FEW_SHOT_MODE
                        to use zero shot mode
  --few_shot_length FEW_SHOT_LENGTH
                        the length of dataset and must be greater than 5
  --few_shot_path FEW_SHOT_PATH
                        the path that the few-shot sequence data is placed
  --save_find_fig SAVE_FIND_FIG
                        whether to save figure when searching the best scale
  --base_scope BASE_SCOPE
                        use the scope when find the best dfg-scale
```

注意：当有使用few-shot数据集作为输入时，记得设置--few_shot_mode=True否则模型无法读取你的few-shot数据集。

参数解释：
```bat
mini_station_num：微站数量
data_path：测试数据地址(xlsx/csv格式)
pt_path：权重文件
dfg_scale：加权残差连接使用的scale值
norm：标准化方法（'max-min'和'standard'两种方式可选）
few_shot_mode：True/False，是否使用few-shot模式挑选scale
few_shot_length：few-shot数据集的数据长度（不小于5）
few_shot_path：few-shot数据集的存放地址
save_find_fig：True/False，是否保存搜索scale时和R^2的关系图
base_scope：搜索最佳scale时范围的上/下界（绝对值）
```



