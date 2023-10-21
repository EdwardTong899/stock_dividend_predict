# stock_dividend_predict 股票填息預測

 
   - 使用2種分類器 (ADAboost, XGBoost) 預測股票在30天內是否可以完成填息  
   - 資料來源https://goodinfo.tw/tw/index.asp
     
# 執行流程
1. 開啟colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l6B7ddo04UVVsFEqxMm4bXFOHaqUAGTe#scrollTo=dzPjAPJQ5qSY)
2. Import data  
    - 2019.csv  
    - 2020.csv  
    - 2021.csv
    - 2022.csv
3. 取得 Feature，共有三類 * 過去三年
    - cac_dividend_rate(dataframe) #計算股息比率
    - stock_dividend(dataframe) #是否有股利 [0,1] 0=沒有, 1=有
    - fill_day(dataframe)      #填息時間

4.選擇分類器
    - XGBoost model  
    - AdaBoost model

6. 預測  
  ```shell
from xgboost import XGBClassifier
xg1 = XGBClassifier()
xg1=xg1.fit(X_train, Y_train)
xg1_val=xg1.predict(X_valid)


accuracy_score(Y_valid,xg1_val)
```

7. 分析結果
  ```shell
# Confusion Matrix
mat = confusion_matrix(Y_valid, xg1_val)
sns.heatmap(mat,square= True, annot=True, cbar= False, fmt='d')
plt.xlabel("predicted value")
plt.ylabel("true value")
plt.show()

# Accuracy
accuracy = metrics.accuracy_score(Y_valid, xg1_val)
print('Valdation accuracy:', accuracy)

# precision, recall, f1-score
#target_names = ['0','1','2','3']
target_names = ['0','1']

print("report:\n",classification_report(Y_valid, xg1_val, target_names=target_names))

print('Total time took: {0}s'.format(time.time()-ts))

accuracy_score(Y_valid,xg1_val)
```
