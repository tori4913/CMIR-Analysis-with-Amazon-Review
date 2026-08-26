import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
# 'your_data_file.csv' 부분을 실제 사용하는 데이터 파일 경로로 변경해주세요.
df = pd.read_csv('your_data_file.csv')

# 종속변수 로그 변환
df['log_helpful_count'] = np.log1p(df['helpful_count'])

# ==========================================
# 2. 정렬 및 유틸리티 함수 정의
# ==========================================
control_vars = [
    'amazon_vine',
    'review_rating',
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
  sorted_list.extend([v for v in var_list if v.startswith('CEIR_Text')])
  sorted_list.extend([v for v in var_list if v.startswith('CEIR_Image')])
  sorted_list.extend([v for v in var_list if v.startswith('Privarcy')])
  # 상호작용항 등 기타 변수 처리
  remaining = [v for v in var_list if v not in sorted_list]
  sorted_list.extend(remaining)
  return sorted_list


# ==========================================
# 3. 논문 채택 최종 변수 설정 (Base Elite + Interaction)
# ==========================================
# 기존 논문에서 최종 검증된 베이스 변수들
base_elite = [
    'days_since_review',
    'media_count',
    'overall_rating',
    'review_length',
    'review_rating',
    'verified_purchase',
    'CEIR_Text_incon_text_rating_discrepancy_score',
    'CEIR_Text_text_specific_detail_claims_count',
    'CEIR_Image_incon_image_rating_discrepancy_score',
    'CEIR_Image_incon_visual_quality_score',
    'Privarcy_image_privacy_biometric_body_percent',
    'Privarcy_image_privacy_biometric_face_percent',
    'Privarcy_image_privacy_context_living_space_percent',
]

# 앞서 탐색된 결정적 이미지 시너지 상호작용항 생성 예시
# (데이터에 해당 원본 변수들이 존재한다는 가정하에 수행됩니다)
v1 = 'CEIR_Image_incon_image_rating_discrepancy_score'
v2 = 'CEIR_Image_incon_visual_quality_score'
inter_name = 'INT_IMG_discrepancy_X_quality'

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
  g_df = g_df.dropna(subset=['log_helpful_count'] + final_model_vars)

  # 독립변수 표준화(Standardization)
  g_df[final_model_vars] = StandardScaler().fit_transform(g_df[final_model_vars])

  # 정렬 함수 적용 및 상수항 추가
  sorted_vars = sort_vars_final(final_model_vars)
  X = sm.add_constant(g_df[sorted_vars])
  y = g_df['log_helpful_count']

  # Robust 표준오차(HC3)를 적용한 OLS 모형 적합
  res = sm.OLS(y, X).fit(cov_type='HC3')

  # 결과 출력
  print(f'\n{'='*20} {name} Regression Result {'='*20}')
  print(res.summary())