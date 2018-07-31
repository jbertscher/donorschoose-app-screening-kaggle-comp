import sys
sys.path.append('../src/custom-modules')
from model_diagnostics import *


def count_vectorise(col, ctv, X_train, X_test=None):
    X_train_col = X_train[col]
    if X_test is None:
        ctv.fit(list(X_train_col.ravel()))
        X_train_col_ctv =  ctv.transform(X_train_col.ravel()) 
        return X_train_col_ctv
    else:
        
        # Fitting Count Vectorizer to both training and test sets
        X_test_col = X_test[col].copy()
        ctv.fit(list(X_train_col.ravel()) + list(X_test_col.ravel()))
        X_train_col_ctv =  ctv.transform(X_train_col.ravel()) 
        X_test_col_ctv = ctv.transform(X_test_col.ravel())
        return X_train_col_ctv, X_test_col_ctv

def predict_text(train, test, col, ctv, mnb, train_filter=None, test_filter=None, df_filter_desc=None, show_model_results=False, fit_model=True):
    if train_filter is None or test_filter is None:
        train_filter = train.index
        test_filter = test.index
    if df_filter_desc is None:
        df_filter_desc_for_colname = ''
        df_filter_desc = ''
    else:
        df_filter_desc_for_colname = '_{0}'.format(df_filter_desc)
        
    # Filter on date (before vs after essay format change)
    train_slice = train.loc[train_filter]
    test_slice = test.loc[test_filter]

    if pd.isnull(train_slice[col]).any() == True:
        return None
    X_train_wordvec, X_test_wordvec = count_vectorise(col, ctv, train_slice, test_slice)
    y_train = train_slice['project_is_approved']

    # Turn off unnecessary setWithCopy warning here
    pd.options.mode.chained_assignment = None  # default='warn'
    
    if fit_model == True:
        train.loc[train_filter,
                  '{0}_proba{1}'.format(col, df_filter_desc_for_colname)] = cross_val_predict(mnb, 
                                                                        X_train_wordvec, 
                                                                        y_train, 
                                                                        cv=3, 
                                                                        method='predict_proba')[:,1]
        mnb.fit(X_train_wordvec, y_train)
        test.loc[test_filter, 
                 '{0}_proba{1}'.format(col, df_filter_desc_for_colname)] = mnb.predict_proba(X_test_wordvec)[:,1]
    
    # Re-enable setWithCopy warning to be safe
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    
    if show_model_results == True:
        print('{0} - {1}:'.format(col, df_filter_desc))
        classification_model_cv_results(X_train_wordvec, y_train, mnb, cv=3)