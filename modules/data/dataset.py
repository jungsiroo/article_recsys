import glob
import os
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix

class DataStorage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.article2idx, self.idx2article, self.idx2user, self.user2idx = self._load_dict()

    def _load_dict(self):
        """
        article2idx, idx2article, idx2user, user2idx
        """

        files = glob.glob(os.path.join(f'{self.cfg.root}/cached/*.pkl'))
        files.sort()
        ret = []

        for path in files:
            base_name = os.path.basename(path)
            var_name = os.path.splitext(base_name)[0]

            with open(path, 'rb') as file:
                data = pickle.load(file)

            ret.append(data)

        return ret
    
    def _load_raw_data(self):
        """
        view_log, article_info, sample
        """

        view_log = pd.read_csv(f'{self.cfg.root}/data/view_log.csv')
        article_info = pd.read_csv(f'{self.cfg.root}/data/article_info.csv')
        sub = pd.read_csv(f'{self.cfg.root}/data/sample_submission.csv')

        view_log['userID'] = view_log['userID'].apply(lambda x:self.user2idx[x])
        view_log['articleID'] = view_log['articleID'].apply(lambda x:self.article2idx[x])
        article_info['articleID'] = article_info['articleID'].apply(lambda x:self.article2idx[x])

        return view_log, article_info, sub
    
    def _load_embedding_cos_sim(self, emb_type):
        """Load Multi-Lingual based article embedding
        Args:
            emb_type: title / content
        Returns:
            np.array
        """
        data = np.loadtxt(f"{self.cfg.root}/cached/{emb_type[0]}_embedding")
        sim = cosine_similarity(data, data)

        return torch.Tensor(sim)

    def load_train_dataset(self):
        view_log, article_info, _ = self._load_raw_data()

        if self.cfg.model == "similarity":
            cos_sim = self._load_embedding_cos_sim(self.cfg.emb_type)
            top5_cos_sim = torch.topk(cos_sim, k=6)

            return [view_log, top5_cos_sim]
        
        if self.cfg.model == "npmi":
            # user-article 상호 매트릭스 생성
            user_article_matrix = view_log.groupby(['userID', 'articleID']).size().unstack(fill_value=0)

            # PMI 계산을 위해 user별 view, total_view 계산
            article_counts = user_article_matrix.sum(axis=0)
            total_views = article_counts.sum()

            # 개별 기사 별 조회 비율 계산
            P_vj = article_counts / total_views 

            # 동시 조회 확률 계산을 위해 co_view_matrix 생성 후 자기 자신은 제외
            co_view_matrix = user_article_matrix.T.dot(user_article_matrix)
            np.fill_diagonal(co_view_matrix.values, 0)  

            # vj, vk 의 동시등장 확률 계산
            P_vj_vk = co_view_matrix / total_views  
            P_vj_vk_matrix = coo_matrix(P_vj_vk)

            # pmi 수식 참고 : https://blog.naver.com/naver_search/222439504418
            pmi_matrix = np.log(P_vj_vk / (P_vj.values[:, None] * P_vj.values[None, :]))
            pmi_matrix = np.nan_to_num(pmi_matrix, nan=0.0, posinf=0.0, neginf=0.0)  
            pmi_sparse_matrix = coo_matrix(pmi_matrix)

            # Normalize 진행 (헤비 유저의 영향 최소화)
            with np.errstate(divide='ignore', invalid='ignore'):  
                npmi_values = pmi_sparse_matrix.data / -np.log(P_vj_vk_matrix.data)

            # log(0)과 같은 INF 처리
            npmi_values = np.nan_to_num(npmi_values, nan=0.0, posinf=0.0, neginf=0.0)

            npmi_sparse_matrix = coo_matrix((npmi_values, (pmi_sparse_matrix.row, pmi_sparse_matrix.col)), 
                                            shape=pmi_sparse_matrix.shape)
                
            user_similarity = cosine_similarity(user_article_matrix)
            item_similarity = npmi_sparse_matrix.toarray()
        
            return [user_article_matrix, user_similarity, item_similarity, view_log]
        
        return [view_log, article_info]


    def postprocess(self, result):
        result['userID'] = result['userID'].apply(lambda x:self.idx2user[x])
        result['articleID'] = result['articleID'].apply(lambda x:self.idx2article[x])

        return result