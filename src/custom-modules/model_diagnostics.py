from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Plot a confusion matrix
def plot_cm(y_actual, y_pred):
    cm = confusion_matrix(y_actual, y_pred)
    fig = plt.figure(figsize = (5,5))
    sns.set(font_scale = 1.25)
    sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = ',.0f', annot_kws = {'size': 14})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show();


# Classification model diagnostics
def classification_model_results(y, y_pred, y_pred_proba, show_report=False, show_cm=False):
    if show_report==True:
        print(classification_report(y, y_pred))
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    print('Area under ROC:', roc_auc_score(y, y_pred_proba));
    if show_cm==True:
        plot_cm(y, y_pred)
    

# Cross-validation model fitting and diagnostics for classification
def classification_model_cv_results(X, y, classifier, cv=10, show_report=False, show_cm=False):
    y_pred = cross_val_predict(classifier, X, y, cv=cv)
    y_pred_proba = cross_val_predict(classifier, X, y, cv=cv, method='predict_proba')
    classification_model_results(y, y_pred, y_pred_proba, show_report, show_cm)


# Hold-out validation model fitting and diagnostics
def classification_model_holdout_results(X, y, prefit_classifier):
    y_pred = prefit_classifier.predict(X)
    y_pred_proba = prefit_classifier.predict_proba(X)
    classification_model_results(y, y_pred, y_pred_proba)


def regression_model_cv_results(X, y, classifier, n_folds):
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=n_folds)
    rmse_list = []
    r2_list = []
    adj_r2_list = []
    for train, test in kf.split(X):
        classifier.fit(X[train], y[train])
        y_pred = classifier.predict(X[test])
        y_test = y[test]
        rmse_list.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        r2_list.append(metrics.r2_score(y_test, y_pred))
        adj_r2_list.append(1 - float(len(y_test)-1)/(len(y_test)-len(classifier.coef_)-1)*(1 - metrics.r2_score(y_test, y_pred)))
    mean_cv_rmse = np.mean(rmse_list)
    mean_cv_r2 = np.mean(r2_list)
    mean_cv_adj_r2 = np.mean(adj_r2_list)
    return {'mean_cv_rmse': mean_cv_rmse,
            'mean_cv_r2': mean_cv_r2,
            'mean_cv_adj_r2': mean_cv_adj_r2}
    

def regression_model_cv_report(X, y, classifier, n_folds):
    results = regression_model_cv_results(X, y, classifier, n_folds)
    mean_cv_rmse = results['mean_cv_rmse']
    mean_cv_r2 = results['mean_cv_r2']
    mean_cv_adj_r2 = results['mean_cv_adj_r2']
    print('Average Root Mean Squared Error: {0:.4}, which is {1:.2%} of the range of y'.format(mean_cv_rmse, mean_cv_rmse/(np.max(y)-np.min(y))))
    print('Average R2: {0:.4}'.format(mean_cv_r2))
    print('Average adjusted R2: {0:.4}'.format(mean_cv_adj_r2))

    
def regression_model_report(X_train, y_train, X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    r2 = metrics.r2_score(y_test, y_test_pred)
    adj_r2 = 1 - float(len(y_test)-1)/(len(y_test)-len(classifier.coef_)-1)*(1 - r2)
    print('Average Root Mean Squared Error: {0:.4}, which is {1:.2%} of the range of y'.format(rmse, rmse/(np.max(y_test)-np.min(y_test))))
    print('Average R2: {0:.4}'.format(r2))
    print('Average adjusted R2: {0:.4}'.format(adj_r2)) 
    
    
def plot_actual_vs_predicted(y, y_pred, xlim=None, ylim=None, figsize=(10,10), legend_loc='best'):
    actual_vs_pred = pd.DataFrame({'actual': np.array(y),
                                   'predicted': np.array(y_pred)})

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.regplot(x='actual' , y='predicted', data=actual_vs_pred, ax=ax)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c='red', linewidth=2, label="y = x ('Perfect' fit would show a straight line')")    
    
    ax.set_ylabel('Predicted', fontsize=15)
    ax.set_xlabel('Actual', fontsize=15)
    ax.set_title('Actual vs Predicted Values', fontsize=20, fontweight='bold').set_position([.5, 1.02])
    
    ax.legend(loc=legend_loc, frameon=False, fontsize=10)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(bottom="off", left="off")
    
    plt.show();
    
    
