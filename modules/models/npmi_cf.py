import pandas as pd

class NPMI_CF:
    def __init__(self, cfg, *dataset):
        self.cfg = cfg

        self.user_article_matrix, \
        self.user_similarity, \
        self.item_similarity, \
        self.view_log = dataset


    # 인기 기사 추출 
    def _top5_article(self, dataframe):
        df_freq = dataframe.value_counts("articleID", ascending=False)[:5]
        return df_freq

   
    def fit(self):
        user_predicted_scores = self.user_similarity.dot(self.user_article_matrix)
        item_predicted_scores = self.user_article_matrix.dot(self.item_similarity)
        predicted_scores = self.cfg.alpha * user_predicted_scores + (1-self.cfg.alpha) * item_predicted_scores
        predicted_scores.columns = self.user_article_matrix.columns

        if self.cfg.view == 'include':
            df_freq = self.view_log.groupby("userID").apply(self._top5_article).reset_index()

            # 각 userID 별로 5개의 행을 가지도록 부족한 행 추가
            user_groups = df_freq.groupby('userID')
            new_rows = []

            for userID, group in user_groups:
                article_list = group['articleID'].to_list() # 각 사용자가 조회한 기사 리스트
                candidate = predicted_scores.loc[userID] # 해당 사용자의 기사별 예측 점수

                article_list = list(set(candidate.index) - set(article_list)) # 예측 점수 데이터에서 사용자가 이미 조회한 기사 제외
                candidate = candidate.loc[article_list].sort_values(ascending=False) # 예측 점수가 높은 순서로 정렬

                for idx in range(max(5 - len(group), 0)):
                    new_rows.append({'userID': userID, 'articleID': candidate.index[idx], 'count': None}) # 부족한 행 만큼 새로운 행 추가

            # 새로운 행을 기존 데이터프레임에 추가
            df_freq = pd.concat([df_freq, pd.DataFrame(new_rows)], axis=0)
            df_freq = df_freq.sort_values(by=['userID'], ignore_index=True)

            top_recommendations = df_freq[["userID", "articleID"]].reset_index(drop=True)

        else:
            recommendations = []
            for idx, user in enumerate(self.user_article_matrix.index):    
                # 해당 사용자의 추천 점수 (높은 점수부터 정렬)
                candidate = predicted_scores.loc[user]
                sorted_indices = candidate.argsort()[::-1]
                top5recommend = [article for article in self.user_article_matrix.columns[sorted_indices]][:5]
                
                for article in top5recommend:
                    recommendations.append([user, article])
                    
            # sample_submission.csv 형태로 DataFrame 생성
            top_recommendations = pd.DataFrame(recommendations, columns=['userID', 'articleID'])
        return top_recommendations
