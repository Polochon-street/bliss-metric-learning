import sqlite3
import json
import os
from xdg.BaseDirectory import xdg_data_home
# Regularization code for the survey (~600 answers).
# Works best with lambdas between 0 and 0.1.
from datetime import datetime
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from scipy.optimize import approx_fprime, check_grad, minimize

from scipy.stats import norm
from scipy.spatial.distance import norm as L2_norm
import numpy as np


database_path = os.path.join(xdg_data_home, 'bliss-rs/songs.db')
con = sqlite3.connect(database_path)
cur = con.cursor()

# Absolutely shameless code
query = cur.execute(
    """
    select song_id, training_triplet.id, 1 as song_number, feature, feature.feature_index from feature
    inner join training_triplet on feature.song_id = training_triplet.song_1_id
    union all
    select song_id, training_triplet.id, 2 as song_number, feature, feature.feature_index from feature
    inner join training_triplet on feature.song_id = training_triplet.song_2_id
    union all
    select song_id, training_triplet.id, 3 as song_number, feature, feature.feature_index from feature
    inner join training_triplet on feature.song_id = training_triplet.odd_one_out_id
    order by training_triplet.id, song_number, feature.feature_index
    """
)
# Copy list so we don't have an iterator
query = [x for x in query]

triplets = []

ids = set(t for _, t, _, _, _ in query)

for id in ids:
    current_triplet_list = [(i, p, f) for i, tid, p, f, _ in query if tid == id]
    song1_features = np.array([f for _, p, f in current_triplet_list if p == 1])
    song2_features = np.array([f for _, p, f in current_triplet_list if p == 2])
    song3_features = np.array([f for _, p, f in current_triplet_list if p == 3])
    triplets.append(np.array([song1_features, song2_features, song3_features]))


def d_metric(x1, x2, L=None):
    return d(L, x1, x2)


def d(L, x1, x2):
    L = L.reshape(len(x1), len(x1))
    sqrd = ((x1 - x2).dot(L.dot(np.transpose(L)))).dot(x1 - x2)
    ret = np.sqrt(sqrd)
    return ret


def grad_d(L, x1, x2):
    L = L.reshape(len(x1), len(x2))
    ret = grad_d_squared(L, x1, x2) / (2 * d(L, x1, x2))
    return ret


def grad_d_squared(L, x1, x2):
    L = L.reshape(len(x1), len(x1))
    grad = 2 * np.outer(x1 - x2, x1 - x2).dot(L)
    return grad.ravel()


# x3 here is the odd thing
def delta(L, x1, x2, x3, sigma, second_batch=False):
    ret = (d(L, x2, x3) - d(L, x1, x2)) / sigma
    if second_batch:
        ret = (d(L, x1, x3) - d(L, x1, x2)) / sigma
    return ret


def grad_delta(L, x1, x2, x3, sigma, second_batch=False):
    ret = (grad_d(L, x2, x3) - grad_d(L, x1, x2)) / sigma
    if second_batch:
        ret = (grad_d(L, x1, x3) - grad_d(L, x1, x2)) / sigma
    return ret


def p(L, x1, x2, x3, sigma, second_batch=False):
    cdf = norm.cdf(delta(L, x1, x2, x3, sigma, second_batch))
    if cdf == 0:
        print(delta(L, x1, x2, x3, sigma, second_batch))
    return norm.cdf(delta(L, x1, x2, x3, sigma, second_batch))


def grad_p(L, x1, x2, x3, sigma, second_batch=False):
    return norm.pdf(delta(L, x1, x2, x3, sigma, second_batch)) * grad_delta(
        L, x1, x2, x3, sigma, second_batch
    )


def log_p(L, x1, x2, x3, sigma, second_batch=False):
    return np.log(p(L, x1, x2, x3, sigma, second_batch))


def grad_log_p(L, x1, x2, x3, sigma, second_batch=False):
    return grad_p(L, x1, x2, x3, sigma, second_batch) / p(
        L, x1, x2, x3, sigma, second_batch
    )


def opti_fun(L, X, sigma, l):
    batch_1 = -sum(np.array([log_p(L, x1, x2, x3, sigma) for x1, x2, x3 in X]))
    batch_2 = -sum(np.array([log_p(L, x1, x2, x3, sigma, True) for x1, x2, x3 in X]))
    return batch_1 + batch_2 + l * np.sum(L**2)


def grad_opti_fun(L, X, sigma, l):
    batch_1 = -np.sum(
        np.array([grad_log_p(L, x1, x2, x3, sigma) for x1, x2, x3 in X]),
        0,
    )
    batch_2 = -np.sum(
        np.array([grad_log_p(L, x1, x2, x3, sigma, True) for x1, x2, x3 in X]),
        0,
    )
    return batch_1 + batch_2 + 2 * l * L


def percentage_preserved_distances(L, X):
    count = 0
    for x1, x2, x3 in X:
        d1 = d(L.ravel(), x1, x2)  # short distance
        d2 = d(L.ravel(), x2, x3)  # long distance
        d3 = d(L.ravel(), x1, x3)  # long distance
        if (d1 < d2) and (d1 < d3):
            count = count + 1
    return count / len(X)


def optimize(L0, X, sigma2, l, method):
    l_dim = len(X[0][0])

    res = minimize(
        opti_fun,
        L0,
        args=(X, sigma2, l),
        jac=grad_opti_fun,
        method=method,
    )
    L = np.reshape(res.x, [l_dim, l_dim])
    return (res, L)

# Methods that converged:
# - L-BFGS-B
# - Newton-CG gave best results on survey_features but took forever
# - SLSQP completes, but ehh results (2% more than normally)
# - trust-constr - no more amelioration than the rest
method = "L-BFGS-B"
X = np.array(triplets)
l_dim = len(X[0][0])
sigma2 = 2
L0 = np.identity(len(X[0][0])).ravel()
L_init = L0

design, test = train_test_split(X, test_size=0.2)

#lambdas = [10, 50, 100, 200, 500, 1000, 2500, 5000]
lambdas = [0., 0.001, 0.01, 0.1, 1, 50, 100, 500, 1000, 5000]

accuracies = [[] for _ in lambdas]
accuracies_euclidean = []

print("Started {}".format(datetime.now()))

kf = KFold(n_splits=5)
rounds = 0
for train_index, test_index in kf.split(X):
    rounds = rounds + 1
    X_train, X_test = X[train_index], X[test_index]
    print("Doing {}th fold...".format(rounds))
    accuracies_euclidean.append(percentage_preserved_distances(L0, X_test))
    print("Euclidean accuracy is {}".format(percentage_preserved_distances(L0, X_test)))
    for i, l in enumerate(lambdas):
        res, L = optimize(L_init, X_train, sigma2, l, method)
        print(f"Optimizing was a success? {res}")
        accuracy = percentage_preserved_distances(L, X_test)
        accuracies[i].append(accuracy)
        print("Done for lambda = {}, accuracy is {}".format(l, accuracy))


mean_accuracies = np.array(
    [np.mean(local_accuracies) for local_accuracies in accuracies]
)

idx = mean_accuracies.argmax()
max_accuracy = mean_accuracies[idx]
l = lambdas[idx]
print("Mean accuracy for euclidean is: {}".format(np.mean(accuracies_euclidean)))
print("Best accuracy is {} for lambda = {}\n".format(max_accuracy, l))

res, L = optimize(L_init, design, sigma2, l, method)

print("At the end of the day:")
print(
    "Accuracy for non-trained metric on the test set: {}".format(
        percentage_preserved_distances(L0, test)
    )
)
print(
    "Accuracy for trained metric on the test set: {}".format(
        percentage_preserved_distances(L, test)
    )
)

res, L_total = optimize(L_init, X, sigma2, l, method)
M = L_total.dot(L_total.transpose())
np.save("L_total", L_total)
np.save("M", M)

config_path = os.path.join(xdg_data_home, "bliss-rs/config.json")
with open(config_path, "r") as f:
    config = json.load(f)

with open(config_path, "w") as f:
    config['m'] = {
        "v": 1,
        "dim": M.shape,
        "data": M.ravel().tolist(),
    }
    json.dump(config, f, indent=2)

# If you want to load M from the config
Loaded_M = np.array(config['m']['data']).reshape(config['m']['dim'])

print(f"Done {datetime.now()}, but was it a success? {res}")
