import shap
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectPercentile, SelectFromModel, RFECV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.tree import BaseDecisionTree
from sklearn.svm import SVR, SVC
from sklearn.ensemble import BaseEnsemble, RandomForestRegressor
import lightgbm as lgb


# tools
def model_evaluation(model: LinearRegression, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    score_train = model.score(x_train, y_train)
    score_test = model.score(x_test, y_test)
    rmse = root_mean_squared_error(model.predict(x_test), y_test)
    num_feature = x_train.shape[1]
    print(f'{'R2_train':10}: {score_train}\n{'R2_test':10}: {score_test}\n{"RMSE":10}: {rmse}')
    return score_train, score_test, rmse, num_feature

def feature_selection_by_method(model: LinearRegression, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, method: object|None = None) -> tuple:
    if type(method) == type(None):
        droped_columns = x_train.columns
        print('Didn\'t reject any future')
    else:
        droped_columns = method
        print(f'selected_feature: {droped_columns.values}')

    model.fit(x_train[droped_columns], y_train)
    return model_evaluation(model, 
                            x_train[droped_columns], 
                            y_train, 
                            x_test[droped_columns], 
                            y_test)

def custom_rfecv(X: np.ndarray|pd.DataFrame, y: np.ndarray, estimator: LinearRegression, importance_func: object, scoring=None, cv=5, min_features_to_select=1, step=1, width=0, verbose=False):
    """
    custom_RFECV：任意の重要度関数（importance_func）を使って変数選択
    
    Parameters:
        X: np.ndarray or pd.DataFrame
        y: np.ndarray
        estimator: 学習器（fit/predict可能なもの）
        importance_func: 関数(estimator, X, y) → feature importances の順番付きarrayを返す
        scoring: 評価関数(y_true, y_pred) → スコア（小さいほど良い、例: MSE）
        cv: int または KFoldインスタンス
        min_features_to_select: 最小で残す特徴量数
        step: 1ステップで除去する特徴数
        width: 許容する誤差の範囲
        verbose: ログ出力

    Returns:
        dict: {
            'support': 選ばれた特徴量のブール配列,
            'ranking': 選ばれた特徴量のランキング配列,
            'scores': 選ばれた特徴量のCVスコア,
            'n_features': 選ばれた特徴量の数
        }
    """
    
    
    # change type
    if type(X) is not np.ndarray:
        X = np.array(X)
    if type(y) is not np.ndarray:
        y = np.array(y)


    # set scoring
    if scoring is None:
        scoring = lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)  # 高いほど良い
    
    
    # set variables
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=0)

    n_features = X.shape[1]
    features = np.arange(n_features)
    ranking = np.ones(n_features, dtype=int)
    cv_features = []
    cv_scores = []


    # eliminate score
    while len(features) >= min_features_to_select:
        if verbose:
            print(f"Features remaining: {len(features)}")

        # caluculate mean cv score
        fold_scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, y_train = X[train_idx][:, features], y[train_idx]
            X_test, y_test = X[test_idx][:, features], y[test_idx]
            
            model = clone(estimator).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_scores.append(scoring(y_test, y_pred))
        
        mean_score = np.mean(fold_scores)
        cv_scores.append(mean_score)
        cv_features.append(features)

        # calculate importances
        model = clone(estimator).fit(X[:, features], y)
        importances = importance_func(model, X[:, features], y)

        # eliminate lowest importance feature
        ranks = np.argsort(importances)
        remove = ranks[:step]
        ranking[features[remove]] = np.max(ranking) + 1
        features = np.delete(features, remove)

        if len(features) <= min_features_to_select:
            break

    # calculate selected features & score & ranking
    final_features_id = [i for i in range(len(cv_scores)) if cv_scores[i]<=min(cv_scores)+width][-1]
    final_features = cv_features[final_features_id]
    model = clone(estimator).fit(X[:, final_features], y)
    final_score = np.mean([scoring(y[test], model.predict(X[test][:, final_features]))
                           for _, test in cv.split(X)])
    cv_scores.append(final_score)

    support = np.zeros(n_features, dtype=bool)
    support[final_features] = True
    ranking[~support] = 2  # 残ったもの: 1, それ以外: 2以上

    return {
        'support': support,
        'ranking': ranking,
        'scores': final_score,
        'n_features': len(features)
    }


# Filter Method
def filter_vif(x_train: pd.DataFrame, vif: int = 5) -> list:
    threshold = (vif-1)/vif
    
    feat_corr = set()
    corr_matrix = x_train.corr()**2
    
    for i in range(x_train.shape[1]):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                feat_name = corr_matrix.columns[i]
                feat_corr.add(feat_name)
    
    columns = x_train.columns
    droped_columns = columns.drop(feat_corr)
    
    return droped_columns

def filter_MI(x_train: pd.DataFrame, y_train: pd.DataFrame, percentile: int = 30) -> list:
    percentile_sel_ = SelectPercentile(mutual_info_regression, percentile=percentile)
    percentile_sel_.fit_transform(x_train, y_train)
    
    droped_columns = x_train.columns[percentile_sel_.get_support()]
    return droped_columns

def filter_ANOVA(x_train: pd.DataFrame, y_train: pd.DataFrame, percentile: int=30) -> list:
    percentile_sel_ = SelectPercentile(f_regression, percentile=percentile)
    percentile_sel_.fit_transform(x_train, y_train)
    
    droped_columns = x_train.columns[percentile_sel_.get_support()]
    return droped_columns


# Wrapper Model
def wrapper_RFECV(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, cv: int=5):
    selector = RFECV(model, cv=cv)
    selector.fit_transform(x_train, y_train)

    droped_columns = x_train.columns[selector.get_support()]
    return droped_columns

def sensitivity_importance(estimator: LinearRegression, X: np.ndarray, y: np.ndarray):
    X_mean = X.mean(axis=0)
    importances = []
    for j in range(X.shape[1]):
        X_temp = np.tile(X_mean, (X.shape[0], 1))
        X_temp[:, j] = X[:, j]
        y_pred = estimator.predict(pd.DataFrame(X_temp))
        model_sensitivity_reg = LinearRegression()
        model_sensitivity_reg.fit(X[:, j].reshape(-1, 1), y_pred)
        importances.append(abs(model_sensitivity_reg.coef_[0]))
    return np.array(importances)

def wrapper_RFECV_sensitivity(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, cv: int=5, score_width: float=0):
    selector = custom_rfecv(x_train, y_train, model, sensitivity_importance, cv=cv, width=score_width)

    droped_columns = x_train.columns[selector['support']]
    return droped_columns

def MI_importance(estimator: LinearRegression, X: np.ndarray, y: np.ndarray):
    MI = mutual_info_regression(X, y)
    importances = pd.Series(MI).values
    return np.array(importances)

def wrapper_RFECV_MI(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, cv: int=5, score_width: float=0):
    selector = custom_rfecv(x_train, y_train, model, MI_importance, cv=cv, width=score_width)

    droped_columns = x_train.columns[selector['support']]
    return droped_columns

def SHAP_importance(estimator: LinearRegression, X: np.ndarray, y: np.ndarray, n_sample: int=100, lr=False):
    if isinstance(estimator, (SVR, SVC)):
        lr = True
        
    X_100 = shap.utils.sample(X, n_sample)
    
    if lr:
        model_SHAP_reg = LinearRegression()
        model_SHAP_reg.fit(X, y)
        explainer = shap.Explainer(model_SHAP_reg.predict, X_100)
        shap_values = explainer(X_100)
    else:
        explainer = shap.Explainer(estimator.predict, X_100)
        shap_values = explainer(X_100)
    importances = ((shap_values.values**2)**0.5).mean(axis=0)
    
    return np.array(importances)

def wrapper_RFECV_SHAP(model: object, x_train: pd.DataFrame, y_train: pd.DataFrame, cv: int=5, score_width: float=0, verbose: bool=False):
    selector = custom_rfecv(x_train, y_train, model, SHAP_importance, cv=cv, width=score_width, verbose=verbose)

    droped_columns = x_train.columns[selector['support']]
    return droped_columns


# Embedded Model
def embbeded_lasso(x_train: pd.DataFrame, y_train: pd.DataFrame, cv: int|None=None):
    model_lasso = LassoCV(cv=cv)
    model_lasso_sel_ = SelectFromModel(model_lasso)
    model_lasso_sel_.fit(x_train, y_train)
    droped_columns = x_train.columns[model_lasso_sel_.get_support()]
    
    return droped_columns

def embbeded_elasticnet(x_train: pd.DataFrame, y_train: pd.DataFrame, cv: int|None=None):
    model_en = ElasticNetCV()
    model_en_sel_ = SelectFromModel(model_en)
    model_en_sel_.fit(x_train, y_train)
    droped_columns = x_train.columns[model_en_sel_.get_support()]
    
    return droped_columns