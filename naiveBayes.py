import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

data_path = "./Data/ratings.dat"
df = pd.read_csv(data_path, header=None, sep="::", engine="python")
df.columns = ["user_id", "movie_id", "rating", "timestamp"]
print(f"Dataset Columns: \n {df.columns}\n")

n_users = df["user_id"].nunique()
n_movies = df["movie_id"].nunique()
print(f"Number of users: {n_users}")
print(f"Number of movies: {n_movies}\n")


def load_user_rating_data(df, n_users, n_movies):
    data = np.zeros([n_users, n_movies], dtype=np.intc)
    movie_id_mapping = {}
    for user_id, movie_id, rating in zip(df["user_id"], df["movie_id"], df["rating"]):
        user_id = int(user_id) - 1
        if movie_id not in movie_id_mapping:
            movie_id_mapping[movie_id] = len(movie_id_mapping)
        data[user_id, movie_id_mapping[movie_id]] = rating
    return data, movie_id_mapping


data, movie_id_mapping = load_user_rating_data(df, n_users, n_movies)
values, counts = np.unique(data, return_counts=True)
for value, count in zip(values, counts):
    print(f"Number of rating {value}: {count}")
print(f"\n{df["movie_id"].value_counts()}\n")

target_movie_id = 2858
X_raw = np.delete(data, movie_id_mapping[target_movie_id], axis=1)
Y_raw = data[:, movie_id_mapping[target_movie_id]]

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

print(f"\nShape of X:{X.shape}")
print(f"Shape of Y:{ Y.shape}\n")

recommend = 3
Y[Y <= recommend] = 0
Y[Y > recommend] = 1

n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f"{n_pos} positive samples and {n_neg} negative samples.\n")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)



clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
prediction_prob = clf.predict_proba(X_test)
print(f"Prediction probability:\n{prediction_prob[0:10]}\n")
prediction = clf.predict(X_test)
print(f"Actual prediction: {prediction[:10]}\n")
accuracy = clf.score(X_test, Y_test)
print(f"The accuracy is: {accuracy*100:.1f}%\n")

# Evaluating classification perfomance.
print(f"Confusion Matrix:\n{confusion_matrix(Y_test, prediction, labels=[0,1])}\n")
print(f"Prediction Score: {precision_score(Y_test, prediction, pos_label=1)}\n")
print(f"Recall Score: {recall_score(Y_test, prediction, pos_label=1)}\n")
print(f"F1 Score (pos 1): {f1_score(Y_test, prediction, pos_label=1)}\n")
print(f"F1 Score (pos 0): {f1_score(Y_test, prediction, pos_label=0)}\n")

report = classification_report(Y_test, prediction)
print(f"\nModel Report: {report}\n")

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0] * len(thresholds), [0] * len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            if y == 1:
                true_pos[i] += 1
            else:
                false_pos[i] += 1
        else:
            break


n_pos_test = (Y_test == 1).sum()
n_neg_test = (Y_test == 0).sum()
true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]
print(
    f"\nComputation area under the receiver operating characteristic curve: {roc_auc_score(Y_test, pos_prob)}\n"
)

""" plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color="darkorange", lw=lw)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show() """

# Tuning the model with cross-validation.
k = 5
k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

smoothing_factor_option = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X, Y):
    X_train_k, X_test_k = X[train_indices], X[test_indices]
    Y_train_k, Y_test_k = Y[train_indices], Y[test_indices]
    for alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record[alpha] = {}
        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train_k, Y_train_k)
            prediction_prob = clf.predict_proba(X_test_k)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test_k, pos_prob)
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)


print(f"\nSmoothing fit prior auc")
for smoothing, smoothing_record in auc_record.items():
    for fit_pior, auc in smoothing_record.items():
        print(f"Smoothing: {smoothing}, Fit prior: {fit_prior}, AUC: {auc/k:.5f}")

clf = MultinomialNB(alpha=2.0, fit_prior=False)
clf.fit(X_train, Y_train)

pos_prob = clf.predict_proba(X_test)[:, 1]
print(f"\nAUC with the best model:", roc_auc_score(Y_test, pos_prob))
