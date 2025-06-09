import numpy as np

# This file contains the definitions for the custom machine learning models.
# By placing them in a separate file, we can ensure that both the training
# script and the main application can import and use them consistently,
# which is crucial for saving and loading models with joblib/pickle.

class CustomSVM:
    def __init__(self, C=1.0, gamma='auto', max_iter=100):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.classifiers = []
        self.classes_ = None
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = None

    def _rbf_kernel(self, X1, X2):
        if self.gamma == 'auto':
            gamma_val = 1.0 / X1.shape[1] if X1.shape[1] > 0 else 1.0
        else:
            gamma_val = self.gamma
        pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2*np.dot(X1, X2.T)
        return np.exp(-gamma_val * pairwise_dists)

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        if n_features == 0:
            self.support_vectors_ = np.array([]).reshape(0,0)
            self.dual_coef_ = np.array([]).reshape(1,0)
            self.intercept_ = 0
            return

        K = self._rbf_kernel(X, X)
        alpha = np.zeros(n_samples)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                Ei = np.dot(alpha * y, K[i]) - y[i]
                if (y[i]*Ei < -0.001 and alpha[i] < self.C) or \
                   (y[i]*Ei > 0.001 and alpha[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    Ej = np.dot(alpha * y, K[j]) - y[j]
                    old_alpha_i, old_alpha_j = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L == H: continue
                    eta = 2*K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0: continue
                    alpha[j] = alpha[j] - y[j]*(Ei - Ej)/eta
                    alpha[j] = max(L, min(H, alpha[j]))
                    if abs(alpha[j] - old_alpha_j) < 1e-5: continue
                    alpha[i] = alpha[i] + y[i]*y[j]*(old_alpha_j - alpha[j])

        sv_idx = alpha > 1e-5
        self.support_vectors_ = X[sv_idx]
        self.dual_coef_ = (alpha[sv_idx] * y[sv_idx]).reshape(1, -1)

        non_bound_sv_idx = sv_idx & (alpha > 1e-5) & (alpha < self.C - 1e-5)
        if np.any(non_bound_sv_idx):
            idx_to_use = non_bound_sv_idx
        elif np.any(sv_idx):
            idx_to_use = sv_idx
        else:
            self.intercept_ = 0
            return

        kernel_for_intercept = self._rbf_kernel(X[idx_to_use], self.support_vectors_)
        preds_without_intercept = np.dot(kernel_for_intercept, self.dual_coef_.T).flatten()
        self.intercept_ = np.mean(y[idx_to_use] - preds_without_intercept)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers = []
        if X.shape[0] == 0 or X.shape[1] == 0:
            self.classes_ = np.array([])
            return
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            classifier_instance = CustomSVM(C=self.C, gamma=self.gamma, max_iter=self.max_iter)
            classifier_instance._fit_binary(X, y_binary)
            self.classifiers.append(classifier_instance)

    def decision_function(self, X):
        if not self.classifiers or X.shape[1] == 0:
            return np.zeros((X.shape[0], len(self.classes_ if self.classes_ is not None else [])))
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls_trainer in enumerate(self.classifiers):
            if cls_trainer.support_vectors_ is None or cls_trainer.support_vectors_.shape[0] == 0 or cls_trainer.support_vectors_.shape[1] != X.shape[1]:
                scores[:, i] = 0
                continue
            K = self._rbf_kernel(X, cls_trainer.support_vectors_)
            scores[:, i] = np.dot(K, cls_trainer.dual_coef_.T).flatten() + cls_trainer.intercept_
        return scores

    def predict(self, X):
        if not self.classifiers or (self.classes_ is not None and len(self.classes_) == 0):
            return np.array([0] * X.shape[0])
        decision_scores = self.decision_function(X)
        if decision_scores.shape[1] == 0:
            return np.array([0] * X.shape[0])
        return self.classes_[np.argmax(decision_scores, axis=1)]


class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if n_samples == 0 or n_features == 0:
            return
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        class_counts = np.bincount(y, minlength=n_classes)
        self.class_log_prior_ = np.log((class_counts + 1e-8) / (n_samples + 1e-8 * n_classes))
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        class_map = {cls_val: i for i, cls_val in enumerate(self.classes_)}
        for cls_val in self.classes_:
            cls_idx = class_map[cls_val]
            X_cls = X[y == cls_val]
            if X_cls.shape[0] == 0:
                self.feature_log_prob_[cls_idx] = np.log((self.alpha) / (n_features * self.alpha))
                continue
            total_count_per_feature = X_cls.sum(axis=0) + self.alpha
            self.feature_log_prob_[cls_idx] = np.log(total_count_per_feature / (total_count_per_feature.sum()))
            
    def predict(self, X):
        if self.feature_log_prob_ is None or self.class_log_prior_ is None or self.classes_ is None:
            return np.array([0] * X.shape[0])
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            return np.array([self.classes_[0] if len(self.classes_) > 0 else 0] * X.shape[0])
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)] 