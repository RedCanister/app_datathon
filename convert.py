import pickle as pkl
import pandas as pd

with open("user_part_0.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'user_part_0.csv')

with open("news_label_0.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv(r'news_label_0.csv')

#with open("file.pkl", "rb") as f:
#    object = pkl.load(f)
    
#df = pd.DataFrame(object)
#df.to_csv(r'file.csv')