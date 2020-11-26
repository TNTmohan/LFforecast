## 需要使用的python库
1. pytorch
2. pandas
3. numpy
4. matplotlib
5. xlwt
    用于导出excel文件
6. chinese_calendar
    用于获得节假日信息，pip install chinesecalendar

## 各文件信息
1. ./data/STLF_DATA_IN_1.xls 数据集
2. ./src/LF_Forecasting.ipynb 数据预处理、模型搭建&训练、结果分析
3. ./src/model.th 最终训练好的模型
4. ./src/predict.py 用于预测某一天的负荷曲线
6. ./img 项目过程中的图片