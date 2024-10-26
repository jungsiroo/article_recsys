import pandas as pd

class SimModel:
    def __init__(self, cfg, *dataset):
        self.cfg = cfg
        self.view_log, self.top5_cos_sim = dataset
    
    def fit(self):
        top5_recommend = []

        for user_id in self.view_log['userID'].unique():
            count = {}
            articles = self.view_log[self.view_log['userID'] == user_id]['articleID'].tolist()

            for article in articles:
                top5 = self.top5_cos_sim.indices[article].tolist()
                for a in top5[1:]:
                    count[a] = count.get(a, 0)+1
            
            most_similar = dict(sorted(count.items(), key=lambda x:-x[1]))

            for article in list(most_similar.keys())[:5]:
                top5_recommend.append([user_id, article])

        return pd.DataFrame(top5_recommend, columns=['userID', 'articleID'])

