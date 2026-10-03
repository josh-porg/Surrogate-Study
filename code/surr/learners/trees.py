"""Tree ensembles — the tabular-ML mainstream.

  rf           random forest; UQ = spread of tree predictions          Breiman 2001
  extratrees   extremely randomised trees (smoother averages)          Geurts et al. 2006
  qrf          quantile regression forest (leaf co-membership weights) Meinshausen 2006
  xgb          XGBoost (default-regularised, small CV grid)            Chen & Guestrin 2016  — Kaggle/tabular standard
  lgbm         LightGBM                                                Ke et al. 2017
  catboost     CatBoost (ordered boosting)                             Prokhorenkova et al. 2018
  ngboost      NGBoost, Normal distribution, tree base learners        Duan et al. 2020
  gbt_committee XGBoost bootstrap committee (the thesis scout's model) Poznański 2026 App. G
"""
from __future__ import annotations

import numpy as np

from .base import Learner, register


@register
class RF(Learner):
    name, family, year, field, ref, uq = "rf", "tree ensemble", 2001, "machine learning", "breiman2001random", True

    def _fit(self, U, z):
        from sklearn.ensemble import RandomForestRegressor
        self._m = RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_features=1.0,
                                        random_state=self.seed, n_jobs=1).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)

    def _predict_std(self, U):
        return np.std([t.predict(U) for t in self._m.estimators_], axis=0)


@register
class ExtraTrees(RF):
    name, year, ref = "extratrees", 2006, "breiman2001random"

    def _fit(self, U, z):
        from sklearn.ensemble import ExtraTreesRegressor
        self._m = ExtraTreesRegressor(n_estimators=300, random_state=self.seed, n_jobs=1).fit(U, z)


@register
class QRF(Learner):
    """Quantile regression forest: prediction = weighted empirical distribution of training targets, weights from
    leaf co-membership averaged over trees. Mean and std reported; quantiles available via `quantile`."""
    name, family, year, field, ref, uq = "qrf", "tree ensemble", 2006, "statistics", "meinshausen2006quantile", True

    def _fit(self, U, z):
        from sklearn.ensemble import RandomForestRegressor
        self._m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=self.seed,
                                        n_jobs=1).fit(U, z)
        self._leaves, self._z = self._m.apply(U), z

    def _weights(self, U):
        L = self._m.apply(U)                                               # (m, T)
        W = np.zeros((len(U), len(self._z)))
        for t in range(L.shape[1]):
            same = L[:, [t]] == self._leaves[None, :, t]
            W += same / same.sum(1, keepdims=True)
        return W / L.shape[1]

    def _predict(self, U):
        return self._weights(U) @ self._z

    def _predict_std(self, U):
        W = self._weights(U); mu = W @ self._z
        return np.sqrt(np.maximum(W @ self._z**2 - mu**2, 0))

    def quantile(self, U, q):
        W = self._weights(np.atleast_2d(U)); o = np.argsort(self._z)
        cw = np.cumsum(W[:, o], axis=1)
        return self._yinv(self._z[o][np.argmax(cw >= q, axis=1)])


def _small_cv(est, grid, U, z, seed):
    from sklearn.model_selection import GridSearchCV
    return GridSearchCV(est, grid, cv=min(5, len(z))).fit(U, z).best_estimator_


@register
class XGB(Learner):
    name, family, year, field, ref = "xgb", "gradient boosting", 2016, "machine learning (tabular)", "chen2016xgboost"

    def _fit(self, U, z):
        import xgboost as xgb
        est = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.9, random_state=self.seed, n_jobs=1)
        self._m = _small_cv(est, {"max_depth": [2, 4, 6]}, U, z, self.seed)

    def _predict(self, U):
        return self._m.predict(U)


@register
class LGBM(Learner):
    name, family, year, field, ref = "lgbm", "gradient boosting", 2017, "machine learning (tabular)", "ke2017lightgbm"

    def _fit(self, U, z):
        import lightgbm as lgb
        est = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, min_child_samples=3, subsample=0.9,
                                subsample_freq=1, random_state=self.seed, n_jobs=1, verbose=-1)
        self._m = _small_cv(est, {"num_leaves": [4, 15, 31]}, U, z, self.seed)

    def _predict(self, U):
        return self._m.predict(U)


@register
class CatBoost(Learner):
    name, family, year, field, ref = "catboost", "gradient boosting", 2018, "machine learning (tabular)", "prokhorenkova2018catboost"

    def _fit(self, U, z):
        from catboost import CatBoostRegressor
        self._m = CatBoostRegressor(iterations=600, depth=4, learning_rate=0.05, random_seed=self.seed,
                                    verbose=False, thread_count=1).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)


@register
class NGBoost(Learner):
    name, family, year, field, ref, uq = "ngboost", "gradient boosting", 2020, "machine learning (probabilistic)", "duan2020ngboost", True

    def _fit(self, U, z):
        from ngboost import NGBRegressor
        from sklearn.tree import DecisionTreeRegressor
        self._m = NGBRegressor(Base=DecisionTreeRegressor(max_depth=3), n_estimators=400, learning_rate=0.03,
                               verbose=False, random_state=self.seed).fit(U, z)

    def _predict(self, U):
        return self._m.predict(U)

    def _predict_std(self, U):
        return self._m.pred_dist(U).params["scale"]


@register
class GBTCommittee(Learner):
    """Bootstrap committee of XGBoost models — the thesis `gbt_committee` scout model, with configurable K."""
    name, family, year, field, ref, uq = "gbt_committee", "gradient boosting", 2026, "Poznański 2026 App. G", "poznanski2026towards", True

    def __init__(self, K=10, **kw):
        super().__init__(**kw); self.K = K

    def _fit(self, U, z):
        import xgboost as xgb
        rng = np.random.default_rng(self.seed); self._ms = []
        for k in range(self.K):
            idx = rng.integers(0, len(z), len(z))
            self._ms.append(xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                             random_state=self.seed + k, n_jobs=1).fit(U[idx], z[idx]))

    def _all(self, U):
        return np.vstack([m.predict(U) for m in self._ms])

    def _predict(self, U):
        return self._all(U).mean(0)

    def _predict_std(self, U):
        return self._all(U).std(0)
