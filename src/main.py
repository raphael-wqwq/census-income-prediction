from sklearn.datasets import fetch_openml
data = fetch_openml(data_id=43898, as_frame=True)

df = data.frame

# création csv original

df.to_csv("adult_brut.csv", index=False)

