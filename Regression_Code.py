import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
# 'your_data_file.csv' 부분을 실제 사용하는 데이터 파일 경로로 변경해주세요.
df = pd.read_csv('your_data_file.csv')

# 종속변수 로그 변환 (Y_r)
df['Y_r'] = np.log1p(df['Helpful_Vote_Count'])

# ==========================================
# 2. 정렬 및 유틸리티 함수 정의
# ==========================================
control_vars = [
    'amazon_vine',
    's_r',
    'review_length',
    'verified_purchase',
    'media_count',
    'overall_rating',
    'total_ratings_count',
    'days_since_review',
]


def sort_vars_final(var_list):
  """표 출력 및 모델링을 위한 변수 정렬 함수"""
  sorted_list = []
  sorted_list.extend([v for v in var_list if v in control_vars])
  sorted_list.extend([v for v in var_list if v.startswith('Text_')])
  sorted_list.extend([v for v in var_list if v.startswith('Image_')])
  sorted_list.extend([v for v in var_list if v.startswith('Privacy_')])
  # 상호작용항 등 기타 변수 처리
  remaining = [v for v in var_list if v not in sorted_list]
  sorted_list.extend(remaining)
  return sorted_list


# ==========================================
# 3. 논문 채택 최종 변수 설정 (Base Elite + Interaction)
# ==========================================
base_elite = [
    'days_since_review',
    'media_count',
    'overall_rating',
    'review_length',
    's_r',
    'verified_purchase',
    'Text_Rating_Incon',
    'Text_Detail_Deficiency',
    'Image_Rating_Incon',
    'Image_Visual_Quality',
    'Privacy_Body_Exposure',
    'Privacy_Face_Exposure',
    'Privacy_Context_Exposure',
]

# 상호작용항 생성 예시
v1 = 'Image_Rating_Incon'
v2 = 'Image_Visual_Quality'
inter_name = 'I_PA'

if v1 in df.columns and v2 in df.columns:
  df[inter_name] = df[v1] * df[v2]
  final_model_vars = base_elite + [inter_name]
else:
  final_model_vars = base_elite

# ==========================================
# 4. 최종 OLS 회귀분석 실행 및 결과 확인
# ==========================================
for name, group in [('Vine (Expert)', 1), ('Non-Vine (General)', 0)]:
  # 분석 대상 데이터 필터링 및 결측치 제거
  g_df = df[df['amazon_vine'] == group].copy()
  g_df = g_df.dropna(subset=['Y_r'] + final_model_vars)

  # 독립변수 표준화(Standardization)
  g_df[final_model_vars] = StandardScaler().fit_transform(g_df[final_model_vars])

  # 정렬 함수 적용 및 상수항 추가
  sorted_vars = sort_vars_final(final_model_vars)
  X = sm.add_constant(g_df[sorted_vars])
  y = g_df['Y_r']

  # Robust 표준오차(HC3)를 적용한 OLS 모형 적합
  res = sm.OLS(y, X).fit(cov_type='HC3')

  # 결과 출력
  print(f'\n{"="*20} {name} Regression Result {"="*20}')
  print(res.summary())
