"""
Predict cancer type from gene expression data using an SVM.

Expects two CSV files downloaded from the Kaggle dataset:
  - data.csv   : samples (rows) x genes (columns), first column = sample ID
  - labels.csv : sample ID -> cancer type (BRCA, KIRC, COAD, LUAD, PRAD)

Update the file paths below to match wherever you saved them.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# ----------------------------------------------------------------------
# Generate fake data lines
# ----------------------------------------------------------------------

def generate_fake_samples(X, n_samples=10, method="shuffled", random_state=42):
    """
    X       : original samples x genes DataFrame
    method  : "uniform"   - pure random noise, no relation to real data
              "gaussian"  - random per-gene, matching each gene's mean/std
              "shuffled"  - real values, but scrambled across genes per row
    """
    rng = np.random.default_rng(random_state)
    n_genes = X.shape[1]

    if method == "uniform":
        fake = rng.uniform(X.values.min(), X.values.max(), size=(n_samples, n_genes))

    elif method == "gaussian":
        means = X.mean(axis=0).values
        stds = X.std(axis=0).values
        fake = rng.normal(loc=means, scale=stds, size=(n_samples, n_genes))
        fake = np.clip(fake, X.values.min(), X.values.max())

    elif method == "shuffled":
        sampled = X.sample(n=n_samples, random_state=random_state).copy().values
        fake = np.array([rng.permutation(sampled[:, g]) for g in range(n_genes)]).T

    else:
        raise ValueError("method must be 'uniform', 'gaussian', or 'shuffled'")

    return pd.DataFrame(
        fake,
        columns=X.columns,
        index=[f"FAKE_{method}_{i:03d}" for i in range(n_samples)]
    )

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
DATA_PATH = "dataset/data.csv"
LABELS_PATH = "dataset/labels.csv"

X = pd.read_csv(DATA_PATH, index_col=0)
y_df = pd.read_csv(LABELS_PATH, index_col=0)

# Align samples between the two files, just in case order differs
X = X.loc[y_df.index]
y = y_df.iloc[:, 0]  # the label column

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} genes.")
print("Class counts:\n", y.value_counts())

# ----------------------------------------------------------------------
# 1.5. Generate fake rows and add them as a new "FAKE" class
#      This must happen BEFORE the variance filter / split / feature
#      selection / scaling / PCA, so those steps all learn to account
#      for the difference between real and fake data.
# ----------------------------------------------------------------------
N_FAKE = 300  # comparable in size to your smallest real class (COAD=78 is much smaller,
              # so 300 gives the model plenty of fake examples to learn from)

fake_uniform = generate_fake_samples(X, n_samples=N_FAKE // 3, method="uniform", random_state=1)
fake_gaussian = generate_fake_samples(X, n_samples=N_FAKE // 3, method="gaussian", random_state=2)
fake_shuffled = generate_fake_samples(X, n_samples=N_FAKE // 3, method="shuffled", random_state=3)

fake_X = pd.concat([fake_uniform, fake_gaussian, fake_shuffled])
fake_y = pd.Series(["FAKE"] * len(fake_X), index=fake_X.index)

# Combine with real data
X = pd.concat([X, fake_X])
y = pd.concat([y, fake_y])

print(f"\nAdded {len(fake_X)} fake rows as a new 'FAKE' class.")
print(f"Updated class counts:\n{y.value_counts()}")

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

from sklearn.calibration import CalibratedClassifierCV

# ----------------------------------------------------------------------
# 6. Train the SVM, with cross-validation on the training set
#    NOTE: SVC's probability=True is deprecated (removed in sklearn 1.11).
#    CalibratedClassifierCV wraps the SVM to produce calibrated
#    probabilities instead, which is the recommended replacement.
# ----------------------------------------------------------------------
base_svm = SVC(kernel="linear", random_state=42)
clf = CalibratedClassifierCV(base_svm, ensemble=False)

cv_scores = cross_val_score(clf, X_train_pca, y_train, cv=5)
print(f"\n5-fold CV accuracy on training set: "
      f"{cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

clf.fit(X_train_pca, y_train)

# ----------------------------------------------------------------------
# 7. Evaluate on the held-out test set
# ----------------------------------------------------------------------
y_pred = clf.predict(X_test_pca)

print(f"\nTest set accuracy: {accuracy_score(y_test, y_pred):.3f}\n")

print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=clf.classes_),
    index=clf.classes_,
    columns=clf.classes_
))

# Also grab confidence on real test rows, for comparison later
test_probs = clf.predict_proba(X_test_pca)
real_confidence = test_probs.max(axis=1).mean()
print(f"\nAverage confidence on REAL test rows: {real_confidence:.3f}")

# ----------------------------------------------------------------------
# 8. Sanity check: can the model recognize data that ISN'T real?
# ----------------------------------------------------------------------


fake_df = generate_fake_samples(X, n_samples=20, method="shuffled", random_state=999)
fake_filtered = var_filter.transform(fake_df)   # reuse fitted filter
fake_sel = selector.transform(fake_filtered)    # reuse fitted selector
fake_scaled = scaler.transform(fake_sel)         # reuse fitted scaler
fake_pca = pca.transform(fake_scaled)            # reuse fitted PCA

fake_probs = clf.predict_proba(fake_pca)
fake_preds = clf.predict(fake_pca)

results = pd.DataFrame(fake_probs, columns=clf.classes_, index=fake_df.index)
results["predicted_label"] = fake_preds
results["max_confidence"] = fake_probs.max(axis=1)

print("\nModel's response to FAKE (shuffled) rows (each number represents the model's confidence):")
print(results)
print(f"\nAverage confidence on FAKE rows: {results['max_confidence'].mean():.3f}")
print(f"Average confidence on REAL test rows: {real_confidence:.3f}")