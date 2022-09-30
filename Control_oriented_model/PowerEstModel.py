# Class for power estimation model
# 
# 
# Developed by Fan Feng at Texas A&M university 
# 
from tabnanny import filename_only
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
# 
import pandas as pd
import numpy as np


### Control-oriented model functions
class PowerEstModel(object):
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
        self.models = {}

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


    def _train_gboost_predictor(self,parmodel={}):
        
        if self.X_sup is None:
            X_dat = self.X_training
        else:
            X_dat = self.X_training

            for i in range(len(self.X_sup)):
                X_dat = X_dat.append(self.X_sup[i])

        N_row,N_col =X_dat.shape
        TrainSet_Ratio = 4/5
        N_train = int(N_row*TrainSet_Ratio)

        #### train pump model
        feature_pump = ['Air_Out_Temp', 'Air_Out_Flowrate', 'ZMAT','DeltaT', 'SensibleHeat','OA_DB','OA_WB','OA_HR']
        X_train_pump,y_train_pump = X_dat[feature_pump].iloc[:N_train,:],X_dat['Pump_Mass_FR'].iloc[:N_train]
        X_test_pump,y_test_pump = X_dat[feature_pump].iloc[N_train:,:],X_dat['Pump_Mass_FR'].iloc[N_train:]
        
        self.models['pump'] = self._train_pump_model(X_train_pump,y_train_pump,X_test_pump,y_test_pump)

        #### train chiller model
        feature_chiller = ['Air_Out_Temp', 'Air_Out_Flowrate', 'ZMAT','DeltaT', 'SensibleHeat','OA_DB','OA_WB','OA_HR']
        X_train_chiller,y_train_chiller = X_dat[feature_chiller].iloc[:N_train,:],X_dat['chiller_power'].iloc[:N_train]
        X_test_chiller,y_test_chiller = X_dat[feature_chiller].iloc[N_train:,:],X_dat['chiller_power'].iloc[N_train:]
        
        self.models['chiller'] = self._train_chiller_model(X_train_chiller,y_train_chiller,X_test_chiller,y_test_chiller)

        return self

    def _train_pump_model(self,X_train,y_train,X_test,y_test):

        param = {
            'eta': 0.3, 
            'max_depth': 3,  
            'objective': 'reg:squarederror',  
            'num_class': 3} 

        steps = 20  # The number of training iterations

        xgb_reg_Pump_FlowRate = xgb.XGBRegressor(
            n_estimators=500,
            reg_lambda=1,
            gamma=0,
            max_depth=3)

        xgb_reg_Pump_FlowRate.fit(X_train, y_train,eval_set = [(X_train, y_train), (X_test, y_test)],verbose=False,early_stopping_rounds=20)
        y_pred = xgb_reg_Pump_FlowRate.predict(X_test)
        print('Pump model $R^2$:   [%.8f]' % (r2_score(y_test, y_pred)))
        return xgb_reg_Pump_FlowRate

    def _train_chiller_model(self,X_train,y_train,X_test,y_test):

        param = {
            'eta': 0.3, 
            'max_depth': 3,  
            'objective': 'reg:squarederror',  
            'num_class': 3} 

        steps = 20  # The number of training iterations

        xgb_reg_chiller_power = xgb.XGBRegressor(
            n_estimators=500,
            reg_lambda=1,
            gamma=0,
            max_depth=3)
        xgb_reg_chiller_power.fit(X_train, y_train,eval_set = [(X_train, y_train), (X_test, y_test)],verbose=False,early_stopping_rounds=50)
        
        y_pred = xgb_reg_chiller_power.predict(X_test)
        print('Chiller model $R^2$:   [%.8f]' % (r2_score(y_test, y_pred)))
        return xgb_reg_chiller_power

    def _Calc_FanPower(self,x):

        ############# Fan properties##############

        FlowRate_design =  1.11
        rho_air = 1.17573
        Tot_eff = 0.9
        PreRise_design = 500


        coef_Fan = [0.306256,-0.67301,4.1433446,-2.7942617]

        frac_flow = x/rho_air/FlowRate_design
        frac_PLR = coef_Fan[0]+ coef_Fan[1]*frac_flow+ coef_Fan[2]*frac_flow**2+ coef_Fan[3]*frac_flow**3
        
        FanPower = frac_PLR*FlowRate_design *PreRise_design/Tot_eff
        
        return FanPower

    def _Calc_PumpPower(self,x):
        ### Power properties
        coef_pump2 = [0,1,0,0]


        Pump2_head = 179352
        Flowrate_Design_pump2  = 4.96*10**(-4)
        Power_Desig_pump2 = 126.73458

        frac_flow = x/1000/Flowrate_Design_pump2
        frac_PLR = coef_pump2[0]+ coef_pump2[1]*frac_flow+ coef_pump2[2]*frac_flow**2+ coef_pump2[3]*frac_flow**3
        
        pumpPower = frac_PLR*Power_Desig_pump2
        return pumpPower

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
        Fan_power = []
        Pump_power = []
        chiller_power = []
        total_Power = []
        Light_power = list(X['Light_rate'])
        for i in range(X.shape[0]):
            # Calculate Fan power
            Fan_Flowrate = X['Air_Out_Flowrate'].iloc[i]
            Fan_power.append(self._Calc_FanPower(Fan_Flowrate))

            # Calculate pump power
            feature_pump = ['Air_Out_Temp', 'Air_Out_Flowrate', 'ZMAT','DeltaT', 'SensibleHeat','OA_DB','OA_WB','OA_HR']
            X_pump= X[feature_pump].iloc[i:i+1]
            Pump_Flowrate = self.models['pump'].predict(X_pump)
            Pump_power.append(self._Calc_PumpPower(Pump_Flowrate))

            # calculate chiller power
            feature_chiller = ['Air_Out_Temp', 'Air_Out_Flowrate', 'ZMAT','DeltaT', 'SensibleHeat','OA_DB','OA_WB','OA_HR']
            X_chiller = X[feature_chiller].iloc[i:i+1]
            chiller_power.append(self.models['chiller'].predict(X_chiller))
            
            tot_P_temp = Fan_power[-1]+Pump_power[-1]+chiller_power[-1]+Light_power[i]
            total_Power.append(tot_P_temp[0])

        return [Fan_Flowrate,Pump_power,chiller_power,total_Power]