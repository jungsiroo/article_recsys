import sys
import warnings
import argparse
import pandas as pd
import numpy as np
from modules.dataloader import load_data
from modules.arguments import get_args
from modules.utils import display_line, seed_fix
from sklearn.metrics.pairwise import cosine_similarity

def run(config):
    display_line('Load Data...')
    view_log_train, article_info, submission = load_data()

    # 사용자-기사 행렬 생성
    user_article_matrix = view_log_train.groupby(['userID', 'articleID']).size().unstack(fill_value=0)

    # 사용자 간의 유사성 계산
    user_similarity = cosine_similarity(user_article_matrix)

    # 추천 점수 계산
    user_predicted_scores = user_similarity.dot(user_article_matrix) / np.array([np.abs(user_similarity).sum(axis=1)]).T

    display_line('Calculating Cosine Similarity...')
    # 이미 조회한 기사 포함해서 추천
    recommendations = []
    for idx, user in enumerate(user_article_matrix.index):    
        # 해당 사용자의 추천 점수 (높은 점수부터 정렬)
        sorted_indices = user_predicted_scores[idx].argsort()[::-1]
        top5recommend = [article for article in user_article_matrix.columns[sorted_indices]][:5]
        
        for article in top5recommend:
            recommendations.append([user, article])

    display_line('Recommendation Saving...') 
    # sample_submission.csv 형태로 DataFrame 생성
    top_recommendations = pd.DataFrame(recommendations, columns=['userID', 'articleID'])

    submission['articleID'] = top_recommendations['articleID']

    submission.to_csv(f'{config.save_path}/{config.model}.csv', index=False)

    display_line('Done!!')


if __name__ == "__main__":
    config = get_args()
    warnings.filterwarnings('ignore')

    seed_fix(config.seed)
    run(config)

