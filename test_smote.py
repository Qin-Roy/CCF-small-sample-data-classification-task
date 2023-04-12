from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE,KMeansSMOTE
from imblearn.combine import SMOTETomek,SMOTEENN
from collections import Counter

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=4,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.01, 0.01, 0.97],
                           class_sep=0.8, random_state=0)
print(sorted(Counter(y).items()))

smote_enn = KMeansSMOTE(random_state=42)
# ,cluster_balance_threshold=0.8
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))

# smote_enn = SMOTEENN(random_state=42)
# X_resampled, y_resampled = smote_enn.fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))


# smote_tomek = SMOTETomek(random_state=42)
# X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# print(sorted(Counter(y_resampled).items()))
