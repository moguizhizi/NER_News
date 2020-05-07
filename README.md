## How to train

1. Download BERT chinese model :  
 ```
 wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip  
 ```


1.	环境
linux+Anaconda3+python3.7+CUDA8.0+cudnn6.0
安装指导: https://blog.csdn.net/qq_38901147/article/details/90049666
2.	数据集
一共分为8类标签
本次标注数据修改规范：
1、organization_name   组织/部门名称 
可以举办活动/会议的均标注为该类实体，如青岛市工信局、青岛海尔集团、某学校、某医院等，标注时将“青岛市工信局”整体标注为实体，不要将青岛市与工信局拆开
2、facilities 设施     
不可以举办活动的标注为设施，如某港口、某码头、生态廊道、某机场、公园等，若名字中包含城市名，需将整体进行标注，如“浙江杭州萧山机场”为一个实体
3、activity_name     活动/事件名称 
会议、宣传活动、新闻发布会等，若名字中包含城市名，需将整体进行标注，如“青岛市服务海尔发展工作专班第一次会议” 为一个实体
4、articals   清单/草案/规划/方案等
如十三五规划、XX实施方案、公告等
5、time 时间     
年份、月份、年月、年月日、年月日时分秒等统一标为时间
6、address  地址
国家、省、市、县、街道等统一标注为地址，类似“我市”、“该市”、“我国”等，不进行标注
7、indicators     指标     
8、data 数据     
标注数据时把单位一并标上，如“200亿元”，注意数据和日期，如“17日”根据前后文判断应标注为哪一类
数据集分为三部分
1、	训练集 实体3322个
2、	验证集 实体460个
3、	测试集 实体710个

详见附件
3.	模型
 

Bert模型：数据预处理
CRF模型：增加label的限制
BiLstm+自注意力模型：通过解决长依赖，提取实体
4.	操作步骤
1	调试参数
第一步：导入数据集和标签文件
进入data_preprocess文件夹下，导入train_data.txt，dev_data.txt，test_data.txt，labels.txt

注意：如果更改数据集，标签，必须严格按照格式进行标注和标签更改

第二步：进入虚拟环境
root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# source activate
(base) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# conda env list
base                  *  /usr/local/anaconda3
tensorflow_py37          /usr/local/anaconda3/envs/tensorflow_py37
tensorflow_py37_gpu      /usr/local/anaconda3/envs/tensorflow_py37_gpu
tensorflow_py37_tf14     /usr/local/anaconda3/envs/tensorflow_py37_tf14

(base) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# conda activate tensorflow_py37_gpu

第三步：数据预处理

(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# cd data_preprocess/
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master/data_preprocess# nohup python -u data_util.py > main.log 2>&1 &

查看日志：
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master/data_preprocess# cat main.log

第四步：执行模型
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# nohup python -u run.py > logs/main.log 2>&1 &
[1] 10147
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master#

注意：默认执行模型Bert+Bilstm+CRF

第五步：查看性能指标
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# cat logs/main.log

processed 21962 tokens with 705 phrases; found: 733 phrases; correct: 471.
accuracy:  93.54%; precision:  64.26%; recall:  66.81%; FB1:  65.51
     activity_name: precision:  40.32%; recall:  43.86%; FB1:  42.02  62
          address: precision:  75.82%; recall:  77.85%; FB1:  76.82  153
         articals: precision:   6.67%; recall:   4.00%; FB1:   5.00  15
             data: precision:  82.48%; recall:  75.84%; FB1:  79.02  137
       facilities: precision:  16.67%; recall:  25.00%; FB1:  20.00  12
       indicators: precision:  20.00%; recall:  28.33%; FB1:  23.45  85
organization_name: precision:  68.42%; recall:  70.48%; FB1:  69.44  171
             time: precision:  81.63%; recall:  87.91%; FB1:  84.66  98
2	实体抽取
第一步：待抽取的文件放置NERdata路径下，并更名为pre_text

第二步：执行terminal_predict.py

(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# nohup python -u terminal_predict.py > logs/predict.log 2>&1 &

查看日志：
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# cat logs/predict.log

第三步：获取结果文件pre_result.csv

root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master/result# ll
total 8
drwxr-xr-x  2 root root 4096 5月   7 11:53 ./
drwxr-xr-x 12 root root 4096 5月   7 11:51 ../
-rw-r--r--  1 root root    0 5月   7 11:53 pre_result.csv
5.	命令总结
1、	Bert模型+CRF模型
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# nohup python -u run.py -crf_only=True > logs/main.log 2>&1 &

2、	Bert模型+BiLstm模型+CRF模型
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# nohup python -u run.py > logs/main.log 2>&1 &

3、	Bert模型+BiLstm模型+自注意+CRF模型
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# nohup python -u run.py -is_add_self_attention=True > logs/main.log 2>&1 &

4、	调整训练集的训练次数
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# nohup python -u run.py -num_train_epochs=20 > logs/main.log 2>&1 &

5、	调整Batchsize
(tensorflow_py37_gpu) root@gpu:/sk/BertModel/BERT-BiLSTM-CRF-NER-master# nohup python -u run.py –batch_size = 10> logs/main.log 2>&1 &

    




## reference: 
+ The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://github.com/kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

+ [https://github.com/zjy-ucas/ChineseNER](https://github.com/zjy-ucas/ChineseNER)

+ [https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)
> Any problem please open issue OR email me(ma_cancan@163.com)
