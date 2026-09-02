"""
Predict cancer type from gene expression data using an SVM.

Expects two CSV files downloaded from the Kaggle dataset:
  - data.csv   : samples (rows) x genes (columns), first column = sample ID
  - labels.csv : sample ID -> cancer type (BRCA, KIRC, COAD, LUAD, PRAD)

Update the file paths below to match wherever you saved them.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "data.csv"
LABELS_PATH = PROJECT_ROOT / "dataset" / "labels.csv"

X = pd.read_csv(DATA_PATH, index_col=0)
y_df = pd.read_csv(LABELS_PATH, index_col=0)

# Reorders the samples in DATA to match LABELS
X = X.loc[y_df.index]
y = y_df.iloc[:, 0]  # the label column

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} genes.")
print("Class counts:\n", y.value_counts())

# ----------------------------------------------------------------------
# 2. Drop genes with zero variance (uninformative)
# ----------------------------------------------------------------------
var_filter = VarianceThreshold(threshold=0.0)
X_filtered = var_filter.fit_transform(X)
print(f"After removing zero-variance genes: {X_filtered.shape[1]} genes remain.")

# ----------------------------------------------------------------------
# 3. Train/test split (stratified so class proportions are preserved)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------------------------
# 4. Feature selection - keep the genes most associated with cancer type
#    (fit ONLY on training data to avoid leaking test info)
# ----------------------------------------------------------------------
K = 1000
selector = SelectKBest(score_func=f_classif, k=K)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)
print(f"Selected top {K} genes by ANOVA F-test.")

# ----------------------------------------------------------------------
# 5. Scale features, then compress with PCA
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

N_COMPONENTS = 50
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA: {N_COMPONENTS} components explain "
      f"{pca.explained_variance_ratio_.sum():.1%} of variance.")

# ----------------------------------------------------------------------
# 6. Train the SVM, with cross-validation on the training set
# ----------------------------------------------------------------------
clf = SVC(kernel="linear", random_state=42)

cv_scores = cross_val_score(clf, X_train_pca, y_train, cv=5)
print(f"\n5-fold CV accuracy on training set: "
      f"{cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

clf.fit(X_train_pca, y_train)

# ----------------------------------------------------------------------
# 7. Evaluate on the held-out test set
# ----------------------------------------------------------------------
y_pred = clf.predict(X_test_pca)

print(f"\nTest set accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=clf.classes_),
    index=clf.classes_,
    columns=clf.classes_
))
