# 유아용품 수요 예측 모델을 활용한 재고 건전성 유지와 상품 추천 모델을 활용한 재구매율 증가
유아용품 판매방식 차별화 및 물품 적기 조달로 매출 향상

## 프로젝트 목표
![image](https://user-images.githubusercontent.com/114542921/208446805-5b494aa1-0c68-47fd-b378-0abed0b250c7.png)

## 사용 언어와 툴
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>  <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/>
#### 현재와 마지막 구매의 날짜 차이가 6개월 이상인 회원 분류
* 데이트타임으로 변환하고 계산하는 것이 손에 익지 않아 헷갈렸다.
```python
# 마지막 구매 기준 중복 제외 
sales_member_product_last = sales_member_product.sort_values(by='구매일').drop_duplicates(['고객번호'], keep = 'last')
len(sales_member_product_last)

# 구매일을 데이트타임으로 형변환
sales_member_product_last['구매일'] = pd.to_datetime(sales_member_product_last['구매일'])

# 현재로부터 6개월 전 날짜 계산
from datetime import datetime
from dateutil.relativedelta import relativedelta

sales_member_product_last['이탈기준일'] = sales_member_product_last.loc[53001,'구매일']- relativedelta(months=6)

# 마지막 구매가 6개월 전인 회원은 이탈로 간주
# 가장 최근 2020-08-07 
# 이탈은 1 비이탈은 0
sales_member_product_last['이탈여부'] = sales_member_product_last['구매일'].apply(lambda x: 1 if (sales_member_product_last['이탈기준일'].iloc[1] - x).days>0 else 0)

# 전체 데이터에서 분리하기 위해 인덱스 추출
index_ex = sales_member_product_last[sales_member_product_last['이탈여부']==1].index
index_no_ex = sales_member_product_last[sales_member_product_last['이탈여부']==0].index

# 분리
sales_member_product_ex = sales_member_product.loc[index_ex]
sales_member_product_no_ex = sales_member_product.loc[index_no_ex]
```

#### 주문이 많은날과 적은날 평균 배송일과 주문량 시각화
* 구매횟수와 배송일의 단위가 너무 차이나서 한 그래프에 그릴 수 없었다.
  * 2중 y축 사용
```python
fig , ax1 = plt.subplots()
fig.set_size_inches(10.5, 8.5)
x=['2019년 10월','2020년 6월']
y=[20180, 5248]

ax1.bar(x,y,color = ['#B8D2FF','#FFEEAE'],width=0.7)
y=[2.368, 2.374]

ax2 = ax1.twinx()
ax2.plot(x,y,color='#FFBB9F', marker='o')

ax1.set_ylabel('구매횟수',size=15)
ax2.set_ylabel('배송일',size=15)


ax1.set_ylim(1000,21000)
ax2.set_ylim(1,3)
ax1.tick_params(axis='both', which='major', labelsize=13)
ax2.tick_params(axis='both', which='major', labelsize=13)
plt.show()

```
![image](https://user-images.githubusercontent.com/114542921/208448290-e1025c48-05e6-4e6b-a7ff-9f1a10829d0a.png)


#### 물품 판매량 시계열 모델링
* 랜덤포레스트로는 정확도가 낮게 나와서 여러 시계열 모델을 사용.
```python
# 트레인/ 테스트 구분
train = subset_t.iloc[:550]
test= subset_t.iloc[550:]

# exponential smoothing in Python
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np

# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(np.array(train['물품판매량']), 
                          initialization_method="estimated").fit()

# Trend
fit2 = Holt(np.array(train['물품판매량']), 
            initialization_method="estimated").fit()

# Exponential trend
fit3 = Holt(np.array(train['물품판매량']),
            exponential=True, 
            initialization_method="estimated").fit()

# Additive damped trend
fit4 = Holt(np.array(train['물품판매량']),
            damped_trend=True, 
            initialization_method="estimated").fit()

# Multiplicative damped trend
fit5 = Holt(np.array(train['물품판매량']),
            exponential=True, 
            damped_trend=True, 
            initialization_method="estimated").fit()
            
## Holt's Winters's method for time series data with Seasonality
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

# additive model for fixed seasonal variation
fit6 = HWES(np.array(train['물품판매량']), 
             seasonal_periods=12, 
             trend='add', 
             seasonal='add').fit(optimized=True, use_brute=True)

# multiplicative model for increasing seasonal variation
fit7 = HWES(np.array(train['물품판매량']), 
             seasonal_periods=12, 
             trend='add', 
             seasonal='mul').fit(optimized=True, use_brute=True)       
             
# 예측
forecast_1 = fit1.forecast(len(test))
forecast_2 = fit2.forecast(len(test))
forecast_3 = fit3.forecast(len(test))
forecast_4 = fit4.forecast(len(test))
forecast_5 = fit5.forecast(len(test))
forecast_6 = fit6.forecast(len(test))
forecast_7 = fit7.forecast(len(test))

# 예측 결과 시각화
plt.figure(figsize=(30,5))
# plt.plot(train['물품판매량'], label='Train')
plt.plot(test['구매일'],test['물품판매량'], label='Test')
plt.plot(test['구매일'],forecast_1, label='f1')

plt.plot(test['구매일'],forecast_2, label='f2')
plt.plot(test['구매일'],forecast_3, label='f3')
plt.plot(test['구매일'],forecast_4, label='f4')
plt.plot(test['구매일'],forecast_5, label='f5')
plt.plot(test['구매일'],forecast_6, label='f6')
plt.plot(test['구매일'],forecast_7, label='f7')

plt.legend()
plt.show()

# 오차 확인
from sklearn import metrics

r1 = np.sqrt(metrics.mean_squared_error(y_test, forecast_1)).tolist()
r2 = np.sqrt(metrics.mean_squared_error(y_test, forecast_2)).tolist()
r3 = np.sqrt(metrics.mean_squared_error(y_test, forecast_3)).tolist()
r4 = np.sqrt(metrics.mean_squared_error(y_test, forecast_4)).tolist()
r5 = np.sqrt(metrics.mean_squared_error(y_test, forecast_5)).tolist()
r6 = np.sqrt(metrics.mean_squared_error(y_test, forecast_6)).tolist()
r7 = np.sqrt(metrics.mean_squared_error(y_test, forecast_7)).tolist()
r_df = pd.DataFrame({'f1': r1, 
                    'f2': r2, 
                    'f3': r3, 
                    'f4': r4, 
                    'f5': r5, 
                    'f6': r6, 
                    'f7': r7}, index=['RMSR'])
r_df             
```

## 개선 사항

* 수동이 아닌 이상치 기준의 최적 값을 구해주는 방법 고민해 보기
* 차원 축소와 클러스터링을 모델링과 연관 지어 사용해 보기
* 분석을 좀 더 자세히 해보고 시각화를 통해 인사이트를 내보기
  * 두가지 관점에서 동시에 분석해 보기


## 상세 내용은 피피티와 코드 파일 참고
> RPA2기_프로젝트6_2조_김민솔.pptx

> Alpaco_Python_Project4_1.ipynb
