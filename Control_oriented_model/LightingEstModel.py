# Class for  lighting estimation model
# 
# 
# Developed by Fan Feng at Texas A&M university 
# 
import pandas as pd
import numpy as np

# xgboost: This is a boosting algorithm
import xgboost as xgb

# 
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score


### Control-oriented model functions
class LightingEstModel(object):
    """
    This class is a model of power estimation model.

    :param X_training: Initial training samples, if available.
    :param y_training: Initial training labels corresponding to initial training samples.

    :param **fit_kwargs: keyword arguments.
    """
    def __init__(self,X_init,parmodel={})-> None:
        ## These dataset stores the parts of operational data that are used to train the zone dynamic predictor. 
        ##

        self.X_operation = X_init  # This dataset will store the operational data at run-time. 
        
        ## These dataset stores the parts of operational data that are used to train the zone dynamic predictor. 
        ##
        self.X_training = None  

        ## These dataset stores some supplementary training data. 
        self.X_sup = None    # A list of dataFrame

        self.model_type = None
        self.models = []

        self._fit_to_known(parmodel)

    def _fit_to_known(self,parmodel={}):
        """
        fit estimator to the operational data and labels provided to it so far.

        :param parmodel: Keyword arguments to be passed to the predict method of the estimator. 
        This is not required because a local parameter tuning will be applied.
        """

        self.X_training = self.X_operation

        if self.model_type is None:
            # determine which model suit this zone
            self.model_type = 'gboost'  # This part will be completed in the future. 

        if self.model_type == 'gboost':
            self._train_gboost_predictor(parmodel)
        else:
            self._train_gboost_predictor(parmodel)

        return self

    def _train_individual_model(self,X,y,parmodel={}):
        
        N_row = X.shape[0]
        TrainSet_Ratio = 4/5
        N_train = int(N_row*TrainSet_Ratio)

        X_train,y_train = X.iloc[:N_train,:],y.iloc[:N_train]
        X_test,y_test = X.iloc[N_train:,:],y.iloc[N_train:]

        xgb_reg_light = xgb.XGBRegressor(
            n_estimators=500,
            reg_lambda=1,
            gamma=0,
            max_depth=3)

        xgb_reg_light.fit(X_train, y_train,eval_set = [(X_train, y_train), (X_test, y_test)],verbose=False,early_stopping_rounds=50)
        y_pred = xgb_reg_light.predict(X_test)
        
        print('Light model $R^2$:   [%.8f]' % (r2_score(y_test, y_pred)))
        return xgb_reg_light

    def _train_gboost_predictor(self,parmodel={}):
        
        if self.X_sup is None:
            X_dat = self.X_training
        else:
            X_dat = self.X_training

            for i in range(len(self.X_sup)):
                X_dat = X_dat.append(self.X_sup[i])

        X = X_dat.iloc[:,:-5]
        for i in range(5):
            y_temp = X_dat.iloc[:,-5+i]
            self.models.append(self._train_individual_model(X,y_temp))

        return self

    def artificial_light_calc(self,x):
        x = x*10  # 10Lumens/watt
        Lux_pred = (0.204*x*100-0.713)/100  # sq meter
        return Lux_pred
        
    def _fit_on_new(self,X,parmodel):
        """
        fit estimator to the given data and labels
        """

        # Step 1: Update the datasets
        self.X_training = X
        self.X_operation = X

        # Step 2: retrain the predictor
        self._fit_to_known(parmodel)

        return self

    def fit(self,X,parmodel={}):
        """
        Interface for the fit method of the predictor. Fits the predictor to the supplied dat, then stores it internally
        for the active learning loop

        :param X: The samples to be fitted
        :param y: The corresponding label
        
        """
        return self._fit_on_new(X,parmodel)

    def add_operational_data(self,X)-> None:
        """
        In the end of each timestep, the operational data will be collected and stored. 
        Then, these collected data will be appended after the current operational data.

        :param X:
        :param y:
        """

        if self.X_operation is None:
            self.X_operation = X
        else:
            self.X_operation = pd.concat([self.X_operation,X],axis = 0)
            

    def add_supplementary_data(self,X)-> None:
            """
            In the end of each timestep, the operational data will be collected and stored. 
            Then, these collected data will be appended after the current operational data.

            :param X:
            :param y:
            """
            ## self.X_sup is a list of supplementary dataset
            if self.X_sup is None:
                self.X_sup, self.y_sup = [],[]
                self.X_sup.append(X)
            else:
                self.X_sup.append(X)

    def teach(self,parmodel={}):
        """
        Retrains the predictor with the operational dataset

        :param parmmodel: Keyword arguments to be passed to the predict method of the estimator. 
        This is not required because a local parameter tuning will be applied.
        """
        self._fit_to_known(parmodel)

    def predict(self, X,horizon):
        """
        Interface with the predict method of the estimator

        :param X: The samples to be predicted.
        :param Horizon: The prediction horizon, and the length of X should be consistent with this. 

        :return: Estimato predictios for X
        """
        Illu_list_p1 = []
        Illu_list_p2 = []
        Illu_list_p3 = []
        Illu_list_p4 = []
        Illu_list_p5 = []
        for i in range(horizon):

            # First point
            Illu_temp = self.models[0].predict(X.iloc[i:i+1,:])
            Illu_list_p1.append(Illu_temp[0])

            # 2 point
            Illu_temp = self.models[1].predict(X.iloc[i:i+1,:])
            Illu_list_p2.append(Illu_temp[0])

            # 3 point
            Illu_temp = self.models[2].predict(X.iloc[i:i+1,:])
            Illu_list_p3.append(Illu_temp[0])
            
            # 4 point
            Illu_temp = self.models[3].predict(X.iloc[i:i+1,:])
            Illu_list_p4.append(Illu_temp[0])

            # 5 point
            Illu_temp = self.models[4].predict(X.iloc[i:i+1,:])
            Illu_list_p5.append(Illu_temp[0])

        return [Illu_list_p1,Illu_list_p2,Illu_list_p3,Illu_list_p4,Illu_list_p5]