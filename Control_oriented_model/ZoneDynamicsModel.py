# Class for zone dynamics model
# 
# 
# Developed by Fan Feng at Texas A&M university 
# 
import pandas as pd
import numpy as np

# xgboost: This is a boosting algorithm
import xgboost as xgb

#
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class ZoneDynamicsModel(object):
    """
    This class is a model of zone dynamic estimation.

    :param X_training: Pandas.DataFrame. Initial training samples, if available.
    :param y_training: Pandas.DataFrame. Initial training labels corresponding to initial training samples.

    :param **fit_kwargs: keyword arguments.
    """
    def __init__(self,X_init,y_init,parmodel={})-> None:

        self.X_operation = X_init  # This dataset will store the operational data at run-time. 
        self.y_operation = y_init
        
        ## These dataset stores the parts of operational data that are used to train the zone dynamic predictor. 
        ##
        self.X_training = None  
        self.y_training = None

        ## These dataset stores some supplementary training data. 
        self.X_sup = None    # A list of dataFrame
        self.y_sup = None  

        self.xgb_regressor = None
        self.ARIMA = None
        self.stackModel = None

        self.model_type = None  # If this building is lightweight or heavyweight.
        self.predictor = None
        self._fit_to_known(parmodel)
        
    def _fit_to_known(self,parmodel={}):
        """
        fit estimator to the operational data and labels provided to it so far.

        :param parmodel: Keyword arguments to be passed to the predict method of the estimator. 
        This is not required because a local parameter tuning will be applied.
        """

        self.X_training = self.X_operation
        self.y_training = self.y_operation

        self.predictor = None

        if self.model_type is None:
            # determine which model suit this zone
            self.model_type = 'gboost'  # This part will be completed in the future. 

        if self.model_type == 'gboost':
            self._train_gboost_predictor(parmodel)
        else:
            self._train_gboost_predictor(parmodel)

        return self

    def _generate_Hist_data(self,y,N_prev):
        """
        
        :param y: y is a  Nx1 dataFrame
        """
        y_re = pd.DataFrame({})
        
        for i in range(N_prev):
            y_re["H{}".format(i+1)] = y.iloc[:].shift(i+1)
        return y_re


    def _train_gboost_predictor(self,parmodel={}):
        
        ## Model training process
        # Step 1: 1st level model

        N_row,N_col = self.X_training.shape
        TrainSet_Ratio = 4/5
        N_prev = 12  #hour

        N_train = int(N_row*TrainSet_Ratio)   

        # Prepare training dataset = supplementary data + training dat
        if self.X_sup is None:
            X_train,y_train = self.X_training.iloc[N_prev:N_train,:],self.y_training.iloc[N_prev:N_train]
            X_test,y_test = self.X_training.iloc[N_train:,:],self.y_training.iloc[N_train:]

            X_2nd = self._generate_Hist_data(self.y_training,N_prev)
            X_2nd_train, X_2nd_test = X_2nd.iloc[N_prev:N_train,:],X_2nd.iloc[N_train:,:]
        else:
            X_train,y_train = self.X_sup[0].iloc[N_prev:,:],self.y_sup[0].iloc[N_prev:]
            X_2nd = self._generate_Hist_data(self.y_sup[0],N_prev)
            X_2nd_train = X_2nd.iloc[N_prev:,:]

            for i in range(len(self.X_sup)-1):
                X_train = X_train.append(self.X_sup[i+1].iloc[N_prev:,:])
                y_train = y_train.append(self.y_sup[i+1].iloc[N_prev:])

                X_2nd_temp = self._generate_Hist_data(self.y_sup[i+1],N_prev)
                X_2nd_train = X_2nd_train.append(X_2nd_temp.iloc[N_prev:,:])

            ##  add training data.
            X_train = X_train.append(self.X_training.iloc[N_prev:N_train,:])
            y_train = y_train.append(self.y_training.iloc[N_prev:N_train])

            X_2nd_temp = self._generate_Hist_data(self.y_training,N_prev)
            X_2nd_train = X_2nd_train.append(X_2nd_temp.iloc[N_prev:N_train,:]) 

            X_2nd_test = X_2nd_temp.iloc[N_train:,:]

            X_test,y_test = self.X_training.iloc[N_train:,:],self.y_training.iloc[N_train:]


        n_folds = 5 ## Number of folds in cross validation

        param = {
            'reg_lambda': 1,
            'n_estimators':500,
            'gamma':0,
            'max_depth': 3}

        xgb_reg_ZoneDyn = xgb.XGBRegressor(**param)

        # fit and predict out-of-fold parts of train set
        Xgb_train_1st = cross_val_predict(xgb_reg_ZoneDyn, 
                                    X_train, y=y_train, 
                                    cv=n_folds, n_jobs=1, 
                                    verbose=0).reshape(-1, 1)

        # Fit on full train set and predict test set once
        _ = xgb_reg_ZoneDyn.fit(X_train, y_train)
        Xgb_test_1st = xgb_reg_ZoneDyn.predict(X_test).reshape(-1, 1)

        # Step 2: 2nd level model
        resi_train = np.array(Xgb_train_1st.reshape(1,-1)[0])-np.array(y_train)
        resi_test = np.array(Xgb_test_1st.reshape(1,-1)[0])-np.array(y_test)

        for i in range(N_prev):
            X_2nd_train['dif{}'.format(i+1)] = X_2nd_train['H{}'.format(i+1)]-pd.DataFrame(Xgb_train_1st,index=X_2nd_train.index).iloc[:,0]
            X_2nd_test['dif{}'.format(i+1)] = X_2nd_test['H{}'.format(i+1)]-pd.DataFrame(Xgb_test_1st,index=X_2nd_test.index).iloc[:,0]

        param = {
            'reg_lambda': 1,
            'n_estimators':500,
            'gamma':0,
            'max_depth': 3}

        xgb_2nd = xgb.XGBRegressor(**param)
        _ = xgb_2nd.fit(X_2nd_train,resi_train,eval_set=[(X_2nd_train,resi_train), (X_2nd_test,resi_test)],verbose=False,early_stopping_rounds=20)
        resi_pred = xgb_2nd.predict(X_2nd_test)

        ## step 3: final predictions
        pred_Final = Xgb_test_1st.reshape(1,-1)[0]- resi_pred

        print('1st level MBE:   [%.8f]' % (mean_absolute_error(y_test, Xgb_test_1st)))
        print('FULL MBE:   [%.8f]' % (mean_absolute_error(y_test, pred_Final)))

        print('1st level R2:   [%.8f]' % (r2_score(y_test, Xgb_test_1st)))
        print('FULL $R^2$:   [%.8f]' % (r2_score(y_test, pred_Final)))

        ### step 4: Train the model on the whole training dataset

        # 1st level
        X_train,y_train = X_train.append(X_test),y_train.append(y_test)
        # fit and predict out-of-fold parts of train set
        Xgb_train_1st = cross_val_predict(xgb_reg_ZoneDyn, 
                                    X_train, y=y_train, 
                                    cv=n_folds, n_jobs=1, 
                                    verbose=0).reshape(-1, 1)

        _ = xgb_reg_ZoneDyn.fit(X_train, y_train)
        # 2nd level
        resi_train = np.array(Xgb_train_1st.reshape(1,-1)[0])-np.array(y_train)
        X_2nd_train =  np.concatenate((X_2nd_train,X_2nd_test))

        _ = xgb_2nd.fit(X_2nd_train,resi_train)

        self.predictor = [xgb_reg_ZoneDyn,xgb_2nd]

        return self

    def _fit_on_new(self,X,y,parmodel):
        """
        fit estimator to the given data and labels
        """

        # Step 1: Update the datasets
        self.X_training, self.y_training = X,y
        self.X_operation, self.y_operation = X,y

        # Step 2: retrain the predictor
        self._fit_to_known(parmodel)

        return self

    def fit(self,X,y,parmodel={}):
        """
        Interface for the fit method of the predictor. Fits the predictor to the supplied dat, then stores it internally
        for the active learning loop

        :param X: The samples to be fitted
        :param y: The corresponding label
        
        """
        return self._fit_on_new(X,y,parmodel)


    def add_operational_data(self,X,y)-> None:
        """
        In the end of each timestep, the operational data will be collected and stored. 
        Then, these collected data will be appended after the current operational data.

        :param X:
        :param y:
        """

        if self.X_operation is None:
            self.X_operation = X
            self.y_operation = y
        else:
            self.X_operation = pd.concat([self.X_operation,X],axis = 0)
            self.y_operation = pd.concat([self.y_operation,y],axis = 0)
    
    
    def add_supplementary_data(self,X,y)-> None:
        """
        In the end of each timestep, the operational data will be collected and stored. 
        Then, these collected data will be appended after the current operational data.

        :param X:
        :param y:
        """teach


        ## self.X_sup is a list of supplementary dataset
        if self.X_sup is None:
            self.X_sup, self.y_sup = [],[]
            self.X_sup.append(X)
            self.y_sup.append(y)
        else:
            self.X_sup.append(X)
            self.y_sup.append(y)


    def teach(self,parmodel={}):
        """
        Retrains the predictor with the operational dataset

        :param parmmodel: Keyword arguments to be passed to the predict method of the estimator. 
        This is not required because a local parameter tuning will be applied.
        """
        self._fit_to_known(parmodel)

    def _predict_NextStep(self, X,y_Hist):
        # 1st level prediction
        pred_1 = self.predictor[0].predict(X)

        # 2nd level prediction

        # generate the dataset for the 2nd level model
        d = {}
        for i in range(len(y_Hist)):
            d['H{}'.format(i+1)] = y_Hist[-i-1]

        for i in range(len(y_Hist)):
            d['dif{}'.format(i+1)] = d['H{}'.format(i+1)] - pred_1
        
        dat_2nd = pd.DataFrame(d)

        pred_2 = self.predictor[1].predict(dat_2nd)
        
        pred_final = pred_1 - pred_2

        return pred_final

    def predict(self, X, horizon):
        """
        Interface with the predict method of the estimator

        :param X: The samples to be predicted.
        :param Horizon: The prediction horizon, and the length of X should be consistent with this. 

        :return: Estimato predictios for X
        """

        N_prev = 12

        if self.X_operation.shape[0]<12:
            print("No enough operational data")
            return

        y_pred_list = []
        y_Hist_init = list(self.y_operation.iloc[-N_prev:])

        for i in range(X.shape[0]):

            # Concatenate Actual value with predictions. 
            y_Hist = y_Hist_init + y_pred_list
            y_Hist = y_Hist[-N_prev:]

            y_pred_temp = self._predict_NextStep(X.iloc[i:i+1,:],y_Hist)  # Keep the X as a DataFrame

            y_pred_list.append(y_pred_temp)

        return  y_pred_list
        



