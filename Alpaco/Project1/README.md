# 간편식 수요 예측 및 공정 최적화를 통한 생산 효율 향상
HMR 제조공정 최적화로 매출 증가


## 프로젝트 목표
![image](https://user-images.githubusercontent.com/114542921/208440085-1e8a63a9-3d94-4075-9888-ed8e6cc729b4.png)

## 사용 언어와 툴
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>  <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/>


#### 품목 코드의 앞자리가 1인 품목과 2인 품목 비교하기
* 데이터프레임의 앞자리만 자르는 방법을 몰라서 [0]처럼 인덱스로 불러오려 했지만 실패.
  * slice를 사용
```python
# 1과 2 나누기
product_s_1 = product_s[product_s['품목코드'].str.startswith('1')==True]
print(len(product_s_1))

# 중복제거
product_s_1_d =  product_s_1.drop_duplicates(subset=['품목코드'], keep='first', ignore_index=False)
print(len(product_s_1_d))

product_s_2 = product_s[product_s['품목코드'].str.startswith('2')==True]
print(len(product_s_2))

product_s_2_d = product_s_2.drop_duplicates(subset=['품목코드'], keep='first', ignore_index=False)
print(len(product_s_2_d))

# 1인 데이터 품목명에 (재) 추가하기
product_s_1_d['재추가']= product_s_1_d['품목명'].apply(lambda x: str(x)+'(재)')

# 2인 데이터를 1로 변경한 컬럼 생성
product_s_2_d['품목코드_통일'] = product_s['품목코드'].str.slice(start=1)
product_s_2_d['품목코드_통일'] = '1'+product_s_2_d['품목코드_통일']

# 2인 데이터의 품목코드_통일 컬럼 기준으로 합치기
product_s_2_d_merge = product_s_2_d.merge(product_s_1_d, how= 'left', left_on = '품목코드_통일', right_on = '품목코드')
product_s_2_d_merge

```
#### 제품군별 출하완료여부 비율 시각화
* 서브플롯 만드는 법이 항상 헷갈렸는데 이번 기회에 확실히 알게 되었다.
```python
fig, ax = plt.subplots(1,3,figsize=(20,10))
x=['출하미완료','출하완료']
y=product_booking_cooking_no_ideal_source_yn['품목명']
ax[0][0].pie(y, labels=x, labeldistance=1.09, autopct='%.2f%%', pctdistance=0.7, textprops= {'fontsize':15},
             colors=palette,wedgeprops={'ec':'w', 'lw':1})

y=product_booking_cooking_no_ideal_bob_yn['품목명']
ax[0][1].pie(y, labels=x, labeldistance=1.09, autopct='%.2f%%', pctdistance=0.7, textprops= {'fontsize':15},
             colors=palette,wedgeprops={'ec':'w', 'lw':1})

y=product_booking_cooking_no_ideal_dress_yn['품목명']
ax[0][2].pie(y, labels=x, labeldistance=1.09, autopct='%.2f%%', pctdistance=0.7, textprops= {'fontsize':15},
             colors=palette,wedgeprops={'ec':'w', 'lw':1})

ax[0][0].set_title('source')
ax[0][1].set_title('bob')
ax[0][2].set_title('dress')

```
![image](https://user-images.githubusercontent.com/114542921/208444809-0076906e-9768-4967-b32e-52e37d4b1928.png)

#### 전날 수량값으로 채운 데이터프레임 만들고 모델링
* 그냥 수치형 컬럼이나 범주형 컬럼을 수치형으로 바꾼 후 인풋으로 넣는 것이 아니라 관련 있는 컬럼만 넣어야 한다.
* 랜덤포레스트, 그래디언트부스팅, 리니어리그레션을 모두 사용해 봤지만 정확도가 높게 나오지 않아서 다음 모델링엔 시계열을 사용해 볼 예정이다.
```python
# 전날 수량값으로 채운 데이터프레임 만들기
def model(x):
    subset_plot_sort_test_model = subset_plot_sort_kg.tail(x) 
    for i in range(x):
        df_shifted = subset_plot_sort_kg.shift(i+1)
        df_shifted = df_shifted.dropna()
        day = df_shifted.tail(x).values
        subset_plot_sort_test_model[f'day{i+1}'] =day
        subset_plot_sort_test_model= subset_plot_sort_test_model.iloc[:,:3].copy()
    return subset_plot_sort_test_model
    
# 인풋, 타겟 설정
x=subset_plot_sort_test_model.drop(['수주수량KG'], axis=1) # input
y=subset_plot_sort_test_model['수주수량KG'] # target

# 트레인/ 테스트 구분
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)

# 학습
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(max_depth=10)
clf.fit(x_train_sc, y_train)

# train 정확도
clf.score(x_train, y_train)
```

## 개선 사항
전처리를 어떤 기준으로 해야 하는지 정하는 게 힘들고 과정 자체도 오래 걸려서 힘들었다.
- 접근 방향성을 잘 잡고 기준을 명확히 한다.

상관도가 높다고 모델이 정확도 높게 만들어지는 것은 아니어서 모델의 인풋 값을 정하는 것이 어려웠다.
- 시각화를 통해 데이터 분석을 충분히 진행한 후 모델링을 해야 한다.

제품군을 분류하는 간단한 방법이 뭐가 있을지 고민해 본다.

## 상세 내용은 피피티와 코드 파일 참고
> RPA2기_프로젝트4_3조_김민솔.pptx

> Alpaco_Python_Project2.ipynb

