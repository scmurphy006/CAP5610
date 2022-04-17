import numpy as np
import pandas as pd
from surprise import SVD, Dataset, KNNBasic, KNNWithMeans, Reader, accuracy
from surprise.model_selection import cross_validate

similarities = ["msd", "cosine", "pearson"]

reader = Reader(sep=",", rating_scale=(0, 5), skip_lines=1)
data = Dataset.load_from_file("ratings_small.csv", reader=reader)

# The equivalent of Probabilistic Matrix Factorization
algo = SVD()
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# User based Collaborative Filtering(KNNWithMeans based on user)
for s in similarities:
    sim_options = {"name": s}
    algo = KNNWithMeans(sim_options=sim_options)
    cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

stats = []

best_k = 0
best_score = 10
for k in range(3, 58, 3):
    algo = KNNBasic(k=k)
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5)
    # Calculate the average of the mean rmse and mae scores
    current_rmse = sum(results["test_rmse"]) / len(results["test_rmse"])
    current_mae = sum(results["test_mae"]) / len(results["test_mae"])
    current_score = (current_rmse + current_mae) / 2
    stats.append([k, current_rmse, current_mae])
    # If current_score < best score, update best_k and best_score
    if current_score < best_score:
        best_score = current_score
        best_k = k

    print("Processed: " + str(k))

print("Best k is " + str(best_k) + ". With an overall average of " + str(best_score))
for i in stats:
    print(i)

# Item based Collaborative Filtering(KNNWithMeans based on item)
for s in similarities:
    sim_options = {"name": s, "user_based": False}
    algo = KNNWithMeans(sim_options=sim_options)
    cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

stats = []

best_k = 0
best_score = 10
for k in range(3, 58, 3):
    sim_options = {"user_based": False}
    algo = KNNBasic(sim_options=sim_options, k=k)
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5)
    # Calculate the average of the mean rmse and mae scores
    current_rmse = sum(results["test_rmse"]) / len(results["test_rmse"])
    current_mae = sum(results["test_mae"]) / len(results["test_mae"])
    current_score = (current_rmse + current_mae) / 2
    stats.append([k, current_rmse, current_mae])
    # If current_score < best score, update best_k and best_score
    if current_score < best_score:
        best_score = current_score
        best_k = k

    print("Processed: " + str(k))

print("Best k is " + str(best_k) + ". With an overall average of " + str(best_score))
for i in stats:
    print(i)
