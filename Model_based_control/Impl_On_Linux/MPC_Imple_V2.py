## Implementation of my MPC framework
## ffeng@tamu.edu


from cProfile import run
import random

import sys,os
import numpy as np
import pandas as pd

import pygad
import pywincalc



# import pyfmi
from pyfmi import load_fmu

#import control-oriented models
from Control_oriented_model import LightingEstModel,PowerEstModel,ZoneDynamicsModel

from os.path import exists


def preProc_ZoneDyn(dat_log,dat_out,dat_wea):
     ## pre-process data
    dat_feature = dat_log
    dat_feature.columns = ['EC_V','B_Height','B_angle']

    dat_feature = dat_feature.join(dat_out['NODE 197:System Node Mass Flow Rate [kg/s](Each Call)'])
    dat_feature = dat_feature.join(dat_out['NODE 197:System Node Temperature [C](Each Call)'])
    dat_feature = dat_feature.join(dat_out['LIGHT_28908:Lights Electricity Rate [W](Each Call)'])

    feature_wea = ['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)',
                'Environment:Site Horizontal Infrared Radiation Rate per Area [W/m2](Hourly)',
                'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Hourly)',
                'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Hourly)',
                'ELEC_28908:Electric Equipment Electricity Rate [W](Hourly)',
                '28908 PEOPLE:People Occupant Count [](Hourly)']
    dat_feature = dat_feature.join(dat_wea[feature_wea])


    ## rename columns
    dat_feature.columns = ['EC_V', 'B_Height', 'B_angle',
                        'SA Mass Flow Rate',
                        'SA System Node Temperature',
                        'Lights Electricity Rate',
                        'Site Outdoor Air Drybulb Temperature',
                        'Site Horizontal Infrared Radiation Rate per Area',
                        'Site Diffuse Solar Radiation Rate per Area',
                        'Site Direct Solar Radiation Rate per Area',
                        'Electric Equipment Electricity Rate',
                        'People Occupant Count']    

    return dat_feature
def penalty_func(ZMAT,Illu_lvl,hyperParam,timeStamps=[0]):
    ## setpoint 
    SP_list = [24]*24 #[26.7]*5+[25.6]+[25]+[24]*15+[26.7]*2 # [18,24]
    SP_ill = [500]*24

    ThermalComfort_range = 0.5
    VisualComfort_range = 50

    residuals_thermal = 0
    residuals_visual = 0

    # placeHolder
    #for i in range(timeStamps.shape[0]):
    #    dtime = timeStamps.iloc[i,0]
    #    hourOfDay = int(dtime.hour)
    #    residuals += max(ZMAT[i]-SP_list[hourOfDay]-ThermalComfort_range,0)


    for i in range(hyperParam['PH']):
        residuals_thermal += max(ZMAT[i]-24-ThermalComfort_range,0)
        residuals_visual += max(Illu_lvl[i]-500-VisualComfort_range,0)

    return residuals_thermal,residuals_visual
def fitness_func(x,solution_idx):
    # this is just a fake function
    return 0
def fitness_function(x,solution_idx,hyperParam):
    ## define the fitness function here. 

    ### run prediction first
    ZMAT, Tot_power, Illu_lvl = run_prediction(x,solution_idx,hyperParam)

    ### define Utility rate   ## placeholder
    uRate = [0.058]*24

    alpha = [10*10,10*8] # This value should be adjusted

    residuals_values = penalty_func(ZMAT,Illu_lvl,hyperParam)

    total_Cost = -sum([0.058*x  for i, x  in enumerate(Tot_power)]) - alpha[0] * residuals_values[0] - alpha[1] * residuals_values[1]
    return total_Cost

def run_prediction(x,solution_idx,hyperParam):
    ## This function run prediction for the incoming horizon

    x = [4,90,100]+x

    ## Step 0: Pre-process inputs for models
    # predHor = hyperParam['PH']
    # tim = hyperParam['tim]
    # controlVariables = [2.5,0,5,350,0.25,14,0] EC_V,blind_angle,blind_height, light_lelvel, SAT_Flowrate, SAT_SP,on-off 
    hourOfYear = int(hyperParam['tim']/3600)

    ZoneDynamicsModel_obj = hyperParam['ZoneDynamicsModel'] 
    PowerEstModel_obj = hyperParam['PowerEstModel'] 
    LightingEstModel_obj = hyperParam['LightingEstModel'] 
    

    ## Step 1: predict

    dat_Op = pd.read_csv(r'./Data/OperationDat.csv')
    # Zone dynamics estimation
    temp_dict_ZoneDyn = {}
    temp_dict_ZoneDyn['EC_V']=x[0:0+hyperParam['PH']]
    temp_dict_ZoneDyn['B_Height']=x[0+hyperParam['PH']:0+hyperParam['PH']*2]
    temp_dict_ZoneDyn['B_angle']=x[0+hyperParam['PH']*2:0+hyperParam['PH']*3]
    temp_dict_ZoneDyn['Lights Electricity Rate']=x[0+hyperParam['PH']*3:0+hyperParam['PH']*4]
    temp_dict_ZoneDyn['SA Mass Flow Rate']=x[0+hyperParam['PH']*4:0+hyperParam['PH']*5]
    temp_dict_ZoneDyn['SA System Node Temperature'] =x[0+hyperParam['PH']*5:0+hyperParam['PH']*6]

    temp_dict_ZoneDyn['Site Outdoor Air Drybulb Temperature'] = list(dat_Op['OA_DB'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    temp_dict_ZoneDyn['Site Horizontal Infrared Radiation Rate per Area'] = list(dat_Op['Site Horizontal Infrared Radiation Rate per Area'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    temp_dict_ZoneDyn['Site Diffuse Solar Radiation Rate per Area'] = list(dat_Op['Site Diffuse Solar Radiation Rate per Area'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    temp_dict_ZoneDyn['Site Direct Solar Radiation Rate per Area'] = list(dat_Op['Site Direct Solar Radiation Rate per Area'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    temp_dict_ZoneDyn['Electric Equipment Electricity Rate'] = list(dat_Op['Electric Equipment Electricity Rate'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    temp_dict_ZoneDyn['People Occupant Count'] = list(dat_Op['People Occupant Count'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    temp_DF_ZoneDyn = pd.DataFrame(temp_dict_ZoneDyn)

    ZMAT_pred = ZoneDynamicsModel_obj.predict(temp_DF_ZoneDyn,hyperParam['PH'])

    ZMAT_pred = [v[0] for v in ZMAT_pred]

    # Power estimation
    dict_PowerEst = {}

    # These are just placeholders
    dict_PowerEst['Date\time'] = [0]*hyperParam['PH']
    dict_PowerEst['Fan_power'] = [0]*hyperParam['PH']
    dict_PowerEst['Pump_power'] = [0]*hyperParam['PH']
    dict_PowerEst['Pump_Mass_FR'] = [0]*hyperParam['PH']
    dict_PowerEst['Air_Out_HR'] = [0]*hyperParam['PH']
    dict_PowerEst['ZMAT_HR'] = [0]*hyperParam['PH']
    
    #
    dict_PowerEst['Light_rate'] = x[0+hyperParam['PH']*3:0+hyperParam['PH']*4]
    dict_PowerEst['Air_Out_Flowrate'] = x[0+hyperParam['PH']*4:0+hyperParam['PH']*5]
    dict_PowerEst['Air_Out_Temp'] = x[0+hyperParam['PH']*5:0+hyperParam['PH']*6]
    dict_PowerEst['ZMAT'] = ZMAT_pred
    dict_PowerEst['OA_DB'] = list(dat_Op['OA_DB'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    dict_PowerEst['OA_WB'] = list(dat_Op['Site Outdoor Air Wetbulb Temperature'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    dict_PowerEst['OA_HR'] = list(dat_Op['Site Outdoor Air Humidity Ratio'].iloc[hourOfYear:hourOfYear+hyperParam['PH']])
    
    DF_PowerEst = pd.DataFrame(dict_PowerEst)
    DF_PowerEst['DeltaT'] = DF_PowerEst['ZMAT']-DF_PowerEst['Air_Out_Temp']
    DF_PowerEst['SensibleHeat'] = DF_PowerEst['DeltaT']*DF_PowerEst['Air_Out_Flowrate']
    Power_pred = PowerEstModel_obj.predict(DF_PowerEst,6)[-1]

    # Lighting estimation
    dat_Op_light = pd.read_csv(r'./Data/TrainData_1y.csv')
    DF_lightEst = dat_Op_light.iloc[hourOfYear:hourOfYear+hyperParam['PH'],1:-5]
    DF_lightEst['angle'] = x[0+hyperParam['PH']*2:0+hyperParam['PH']*3]
    height_temp = x[0+hyperParam['PH']:0+hyperParam['PH']*2]
    DF_lightEst['height'] = [temp*10 for temp in height_temp]
    DF_lightEst['ec'] = x[0:0+hyperParam['PH']]

    Ill_p1_list,Ill_p2_list,Ill_p3_list,Ill_p4_list,Ill_p5_list = LightingEstModel_obj.predict(DF_lightEst,hyperParam['PH'])

    # calculate artificial lighting
    light_lvl = x[0+hyperParam['PH']*3:0+hyperParam['PH']*4]
    Ill_total = []
    Ill_art = []
    for i in range(hyperParam['PH']):
        Ill_art.append(LightingEstModel_obj.artificial_light_calc(light_lvl[i])) 
        Ill_total.append(Ill_art[i]+Ill_p2_list[i])
    
    #
    print("prediction completed")
    return ZMAT_pred,Power_pred,Ill_total
    #

def init_models():
    print("===========Init control-oriented models===========")
    ###### Init zone dynamics model ######
    # 0. read data
    dat1 = pd.read_excel(r'./Data/Exp2_Jul1_Jul31.xlsx')
    dat2 = pd.read_csv(r'./Data/log.csv',header = None)

    dat_wea = pd.read_excel(r'./Data/wea_dat.xlsx')

    dat_ZoneDyn = preProc_ZoneDyn(dat2,dat1,dat_wea)

    ## 1. Initialize
    y = dat1['28908 THERMAL ZONE:Zone Mean Air Temperature [C](Each Call)']
    y.name = 'label_ZMAT'
    ZoneDynamicsModel_obj = ZoneDynamicsModel.ZoneDynamicsModel(dat_ZoneDyn.iloc[:,:],dat1['28908 THERMAL ZONE:Zone Mean Air Temperature [C](Each Call)'].iloc[:])

    ## 2. add a supplementary dataset
    ## read data
    dat1 = pd.read_excel(r'./Data/Sup_dat/Jul1_Jul14.xlsx')
    dat2 = pd.read_csv(r'./Data/Sup_dat/log.csv',header = None)

    dat_wea = pd.read_excel(r'./Data/Sup_dat/wea_dat.xlsx')

    dat_sup = preProc_ZoneDyn(dat2,dat1,dat_wea)

    y_sup = dat1['28908 THERMAL ZONE:Zone Mean Air Temperature [C](Hourly)']
    y_sup.name = 'label_ZMAT'
    ZoneDynamicsModel_obj.add_supplementary_data(dat_sup,y_sup)
    ZoneDynamicsModel_obj.teach()


    ###### Init power estimation model ######
    print("power estimation model")
    ## read data
    dat = pd.read_csv(r'./Data/PowerEstData.csv',header = 0)

    ## 1. Initialize
    PowerEstModel_obj = PowerEstModel.PowerEstModel(dat.iloc[:496,:])

    ## 2. add a supplementary dataset
    ## read data
    PowerEstModel_obj.add_supplementary_data(dat.iloc[496:,:])
    PowerEstModel_obj.teach()

    ######### Init lighting estimation model ######

    ## read data
    dat = pd.read_csv(r'./Data/TrainData_1y.csv',header = 0)

    ## 1. Initialize
    LightingEstModel_obj = LightingEstModel.LightingEstModel(dat.iloc[:-200,1:])

    ## 2. add a supplementary dataset
    ## read data
    LightingEstModel_obj.add_supplementary_data(dat.iloc[-200,1:])
    LightingEstModel_obj.teach()

    return ZoneDynamicsModel_obj,PowerEstModel_obj,LightingEstModel_obj

def flatten_list(l):
    NRow= len(l)
    NCol = len(l[0])
    re = []
    for i in range(NRow):
        re = re+l[i]
    return re
def read_Glazing_txt(fileName):
    ## read 
    re_Array = []
    with open(r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\GlazingData"+"\\" + fileName) as fp:
        count  = 0
        for line in fp.readlines():
            count += 1
            if count >= 14:
                re_Array.append([float(x) for x in line.split('\t')])
    return re_Array

def geneGlazingFile(voltage_input):
    if voltage_input >4 or voltage_input <0:
        return 0
    else:
        fileName_Suffix = ["0o0","0o5","1o0","1o5","2o0","2o5","3o0","3o5","4o0","4o0"]

        voltage_low_idx = int(voltage_input/0.5)
        voltage_high_idx = int(voltage_input/0.5)+1
        
        voltage_low = voltage_low_idx * 0.5
        voltage_high = voltage_high_idx *0.5

        voltage_low_FileName = "aIGDB_NbTiO2_n{}V_v3.txt".format(fileName_Suffix[voltage_low_idx])
        voltage_high_FileName = "aIGDB_NbTiO2_n{}V_v3.txt".format(fileName_Suffix[voltage_high_idx])
        
        # read both upper and lower boundaries
        voltage_low_DF = read_Glazing_txt(voltage_low_FileName)
        voltage_high_DF = read_Glazing_txt(voltage_high_FileName)
        
        # create an empty list
        voltage_DF_new = []
        
        # interpolate
        for i in range(len(voltage_high_DF)):
            temp = [0]*4
            if voltage_input>voltage_low:
                temp[0] = voltage_high_DF[i][0]
                temp[1] = (voltage_high_DF[i][1]-voltage_low_DF[i][1])/0.5*(voltage_input-voltage_low)+voltage_low_DF[i][1]
                temp[2] = (voltage_high_DF[i][2]-voltage_low_DF[i][2])/0.5*(voltage_input-voltage_low)+voltage_low_DF[i][2]
                temp[3] = (voltage_high_DF[i][3]-voltage_low_DF[i][3])/0.5*(voltage_input-voltage_low)+voltage_low_DF[i][3]
            else:
                temp[0] = voltage_low_DF[i][0]
                temp[1] = voltage_low_DF[i][1]
                temp[2] = voltage_low_DF[i][2]
                temp[3] = voltage_low_DF[i][3]
            voltage_DF_new.append(temp)
        
        # add a header part
        lines = ["{ Units, Wavelength Units } SI Microns",
                "{ Thickness } 4",
                "{ Conductivity } 1",
                "{ IR Transmittance } TIR=0",
                "{ Emissivity, front back } Emis= 0.84 0.84",
                "{ }",
                "{ Product Name: NbTiO2_" + str(voltage_input)+"V }",
                "{ Manufacturer: Iowa State University }",
                "{ Type: Monolithic }",
                "{ Material: Glass }",
                "{ Appearance: Clear }",
                "{ NFRC ID: 3000003 }",
                "{ Acceptance: # }"]
        
        for i in range(len(voltage_DF_new)):
            lines.append("\t".join(["{0:.4f}".format(x) for x in voltage_DF_new[i]]))
        for i in range(len(lines)):
            lines[i] = lines[i]+"\n"
            
        # write data
        with open(r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\GlazingData\temp.dat",'w+') as fp:
            fp.writelines(lines)
    return lines

def calculate_BSDF(EC_voltage,blind_height,bling_angle):
    # Another part to balabala

    optical_standard_path = r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\standards\W5_NFRC_2003.std"
    optical_standard = pywincalc.load_standard(optical_standard_path)

    glazing_system_width = 4.0772  # width of the glazing system in meters
    glazing_system_height = 0.28435/2  # height of the glazing system in meters

    
    # Define the gap between the shade and the glazing
    gap_1 = pywincalc.Gap(pywincalc.PredefinedGasType.AIR, .0127)  # .0127 is gap thickness in meters

    bsdf_hemisphere = pywincalc.BSDFHemisphere.create(pywincalc.BSDFBasisType.FULL)
    # The BSDF data is currently stored as XML on igsdb.lbl.gov.  As a result it needs to be
    # parsed using the xml string parser instead of the json parser
    if bling_angle>=0:
        bsdf_shade = pywincalc.parse_bsdf_xml_file(r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\tmx\blinds_Angle{}.xml".format(bling_angle))
    else:                                            
        bsdf_shade = pywincalc.parse_bsdf_xml_file(r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\tmx\blinds_Angle_N{}.xml".format(0-bling_angle))
    
    
    ## Generate glazing data file for Voltage to temp.dat
    geneGlazingFile(EC_voltage) 
    
    clear_3_path = r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\GlazingData\temp.dat"
    clear_3 = pywincalc.parse_optics_file(clear_3_path)

    
    # Create a glazing system using the NFRC U environment in order to get NFRC U results
    # U and SHGC can be caculated for any given environment but in order to get results
    # The NFRC U and SHGC environments are provided as already constructed environments and Glazing_System
    # defaults to using the NFRC U environments
    glazing_system_u_environment = pywincalc.GlazingSystem(optical_standard=optical_standard,
                                                        solid_layers=[clear_3, 
                                                                        bsdf_shade],
                                                        gap_layers=[gap_1],
                                                        
                                                        
                                                        width_meters=glazing_system_width,
                                                        height_meters=glazing_system_height,
                                                        environment=pywincalc.nfrc_u_environments(),
                                                        bsdf_hemisphere=bsdf_hemisphere)


    glazing_system_shgc_environment = pywincalc.GlazingSystem(optical_standard=optical_standard,
                                                            solid_layers=[clear_3, bsdf_shade],
                                                            gap_layers=[gap_1],
                                                            width_meters=glazing_system_width,
                                                            height_meters=glazing_system_height,
                                                            environment=pywincalc.nfrc_shgc_environments(),
                                                            bsdf_hemisphere=bsdf_hemisphere)
    
    ## Absorptance matrix
    result_sol = glazing_system_u_environment.optical_method_results("SOLAR",0, 0) 
    
    fAbs_layer1 = result_sol.layer_results[0].front.absorptance.total_direct[2:2+145]
    bAbs_layer1 = result_sol.layer_results[0].back.absorptance.total_direct[2:2+145]
    
    fAbs_layer2 = result_sol.layer_results[1].front.absorptance.total_direct[2+145:2+290]
    bAbs_layer2 = result_sol.layer_results[1].back.absorptance.total_direct[2+145:2+290]
    
    ## Transmittance & Reflectance
    result_sys = result_sol.system_results
    TfSol = flatten_list(result_sys.front.transmittance.matrix)
    RbSol = flatten_list(result_sys.back.reflectance.matrix)

    # visible results
    result_vis = glazing_system_u_environment.optical_method_results("PHOTOPIC",0, 0)
    result_sys = result_vis.system_results
    
    TfVis = flatten_list(result_sys.front.transmittance.matrix)
    RbVis = flatten_list(result_sys.back.reflectance.matrix)

    print("calculated completed")
    return TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2

def calculate_BSDF_clear(EC_voltage,blind_height,bling_angle):
    # Another part to balabala

    optical_standard_path = r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\standards\W5_NFRC_2003.std"
    optical_standard = pywincalc.load_standard(optical_standard_path)

    glazing_system_width = 4.0772  # width of the glazing system in meters
    glazing_system_height = 0.28435/2  # height of the glazing system in meters

    
    # Define the gap between the shade and the glazing
    gap_1 = pywincalc.Gap(pywincalc.PredefinedGasType.AIR, .0127)  # .0127 is gap thickness in meters

    bsdf_hemisphere = pywincalc.BSDFHemisphere.create(pywincalc.BSDFBasisType.FULL)

    print("bp0")
    ## Generate glazing data file for Voltage to temp.dat
    geneGlazingFile(EC_voltage) 
    
    clear_3_path = r"D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Imple_On_HPC\examples\GlazingData\temp.dat"
    clear_3 = pywincalc.parse_optics_file(clear_3_path)

    print("bp1")
    # Create a glazing system using the NFRC U environment in order to get NFRC U results
    # U and SHGC can be caculated for any given environment but in order to get results
    # The NFRC U and SHGC environments are provided as already constructed environments and Glazing_System
    # defaults to using the NFRC U environments
    glazing_system_u_environment = pywincalc.GlazingSystem(optical_standard=optical_standard,
                                                        solid_layers=[clear_3],
                                                        width_meters=glazing_system_width,
                                                        height_meters=glazing_system_height,
                                                        environment=pywincalc.nfrc_u_environments(),
                                                        bsdf_hemisphere=bsdf_hemisphere)

    print("bp2")
    glazing_system_shgc_environment = pywincalc.GlazingSystem(optical_standard=optical_standard,
                                                            solid_layers=[clear_3],
                                                            width_meters=glazing_system_width,
                                                            height_meters=glazing_system_height,
                                                            environment=pywincalc.nfrc_shgc_environments(),
                                                            bsdf_hemisphere=bsdf_hemisphere)

    ## Absorptance matrix
    result_sol = glazing_system_u_environment.optical_method_results("SOLAR",0, 0) 
    fAbs_layer1 = result_sol.layer_results[0].front.absorptance.total_direct[1:1+145]
    bAbs_layer1 = result_sol.layer_results[0].back.absorptance.total_direct[1:1+145]
  

    ## Transmittance & Reflectance
    result_sys = result_sol.system_results
    TfSol = flatten_list(result_sys.front.transmittance.matrix)
    RbSol = flatten_list(result_sys.back.reflectance.matrix)
   

    # visible results
    result_vis = glazing_system_u_environment.optical_method_results("PHOTOPIC",0, 0)
    result_sys = result_vis.system_results
    
    TfVis = flatten_list(result_sys.front.transmittance.matrix)
    RbVis = flatten_list(result_sys.back.reflectance.matrix)

    print("calculated completed")
    return TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1

class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool,hyerParam
        pop_fitness = pool.starmap(fitness_function, [(individual,i,hyperParam) for i,individual in enumerate(self.population)])
        #print(pop_fitness)
        pop_fitness = list(pop_fitness)
        return pop_fitness

if __name__ == "__main__":
    ##--------------Step 0: simulation parameters ------------------ ##
    start_time=60*60*24*184  # start time 
    final_time=60*60*24*186

    hyperParam = {}
    hyperParam['PH'] = 1 # 1hr
    hyperParam['tim'] = 0

    controlVariable = [2.5,90,10,350,0.25,14,0]

    EC_voltage,blind_height,blind_Angle = 4,10,90
    
    TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear =  calculate_BSDF_clear(EC_voltage,blind_height,blind_Angle)
    temp_lines = []
    temp_lines.append(",".join([str(x) for x in controlVariable]) + "\n")
    for temp in [TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear]:
        temp_lines.append(",".join([str(round(x,4)) for x in temp]) + "\n")
    with open(r'BSDF_clear.txt','w') as fp:
        fp.writelines(temp_lines)
    
    
    TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2 =  calculate_BSDF(EC_voltage,blind_height,blind_Angle)
    
    temp_lines = []
    temp_lines.append(",".join([str(x) for x in controlVariable]) + "\n")
    for temp in [TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2]:
        temp_lines.append(",".join([str(round(x,4)) for x in temp]) + "\n")
    with open(r'BSDF_Shade.txt','w') as fp:
        fp.writelines(temp_lines)
    ##--------------Step 1: Initialize control-oriented models ------------------ ##
    print("------Initialize control-oriented models-----------")
    ZoneDynamicsModel_obj,PowerEstModel_obj,LightingEstModel_obj = init_models()
    hyperParam['ZoneDynamicsModel'] = ZoneDynamicsModel_obj
    hyperParam['PowerEstModel'] = PowerEstModel_obj
    hyperParam['LightingEstModel'] = LightingEstModel_obj
    
    # Initial dataset for control-oriented models
    ZoneDynaData = ZoneDynamicsModel_obj.X_training.iloc[-1:,:]
    powerEstData = PowerEstModel_obj.X_training.iloc[-1:,:]
    lightEstData = LightingEstModel_obj.X_training.iloc[-1:,:]

    
    ##--------------Step 2: Initialize e+ FMU simulation  ------------------ ##
    api = "2"
    fmu_path = r'D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\EplusModel\in_WithCFS.fmu'
    model="model_1_"+api
    model = load_fmu(fmu=fmu_path,log_level=4)
    ZMAT_trend = []

    #time_step=900
    if (api=="1"):
        print('1')
        print(('Running FMU file={!s}, API={!s}.'.format(fmu_path, api)))
        #model.setup_experiment(False, 0, 0, True, final_time)
        model.initialize(0, final_time)
    elif(api=="2"):
        print('1')
        print(('Running FMU file={!s}, API={!s}.'.format(fmu_path, api)))
        model.setup_experiment(False, 0,start_time,  True, final_time)
        model.initialize(start_time, final_time)
    else:
        print ('Only FMI version 1.0 and 2.0 are supported')
        exit(1)
    print("------Initialize FMU models completed-----------")

    tim = start_time
    # warm up: 1day is used to collect historical data for zone dynamics model
    warmup_time = start_time + 60*60*39 
    while tim < final_time:
        ## Update hyperParam
        hyperParam['tim'] = tim


        controlVariable = [2.5,0,5,350,0.25,14,0]  
        if tim < warmup_time:
            controlVariable[-1] = 0 # Turn off control 
        else: # implement MPC controller 
            controlVariable[-1] = 1 # Turn on control 

            ######## MPC controller implementation#########
            print("===========MPC start===========")
            
            # Optimization algorithm setting
            num_generations = 30
            sol_per_pop = 50   # Number of individuals

            num_parents_mating = 24
            num_genes = hyperParam['PH']*6

            # define the gene spaces

            #[2.5,0,5,350,0.25,14,0]  
            gene_space =[{'low':0,'high':4}] * hyperParam['PH']+ [range(-90,90)] * hyperParam['PH'] +[range(11)] * hyperParam['PH'] +[{'low':0,'high':400}] * hyperParam['PH']+[{'low':0,'high':0.35}] * hyperParam['PH'] +[{'low':12,'high':18}] * hyperParam['PH']
            
            parent_selection_type = "sss"
            keep_parents = 1

            crossover_type = "single_point"

            mutation_type = "random"
            mutation_num_genes = 1

            ga_instance = PooledGA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_func, # Actually this is not used.
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    gene_space= gene_space,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    mutation_num_genes=mutation_num_genes
                    )

            with Pool(8) as pool:
                print("Start Optimization")

                ga_instance.run()

                print("Op completed")
                print(ga_instance.best_solution())  

                print("all mpi process join again then, and MPC end")

                re = [ga_instance.best_solution(),ga_instance.best_solutions_fitness]


            controlVariable = list(re[0][0])+[1] # Turn on control 
            #write results
            if exists(r'Best_solution_fitness.txt'):
                os.remove(r'Best_solution_fitness.txt')
                
            with open(r'Best_solution_fitness.txt','a+') as fp:
                lines = ','.join([str(tr) for tr in re[1]])
                fp.writelines(lines+"\n")
                
            if exists(r'Best_solution.txt'):
                os.remove(r'Best_solution.txt')  
            with open(r'Best_solution.txt','a+') as fp:
                lines = ','.join([str(tr) for tr in re[0][0]])
                fp.writelines(lines+"\n")
            print(re[0])
        
        ## write the control actions 
        with open(r'D:\Python4Eplus\controlVariables.txt','w') as fp:
            controlVariables = ",".join([str(cV) for cV in controlVariable])
            print("Control signal is :", controlVariables)
            fp.writelines(controlVariables)

        model.do_step(tim, 3600, True)
        ZMAT_temp = model.get('TRoo')
        ZMAT_DF = pd.Series(ZMAT_temp)
        ZMAT_DF.name = 'label_ZMAT'
        
        # Update ZoneDyn model
        hourOfYear = int(hyperParam['tim']/3600)
        temp_dict_ZoneDyn = {}
        dat_Op = pd.read_csv(r'D:\Users\fengf\Onedrive\Google Drive\TAMU\1_Project\NSF\2_Model\3_MPCFramework\Data\OperationDat.csv')

        temp_dict_ZoneDyn['EC_V']=[controlVariable[0]]
        temp_dict_ZoneDyn['B_Height']=[controlVariable[1]]
        temp_dict_ZoneDyn['B_angle']=[controlVariable[2]]
        temp_dict_ZoneDyn['Lights Electricity Rate']=[controlVariable[3]]
        temp_dict_ZoneDyn['SA Mass Flow Rate']=[controlVariable[4]]
        temp_dict_ZoneDyn['SA System Node Temperature'] =[controlVariable[5]]

        temp_dict_ZoneDyn['Site Outdoor Air Drybulb Temperature'] = [dat_Op['OA_DB'].iloc[hourOfYear]]
        temp_dict_ZoneDyn['Site Horizontal Infrared Radiation Rate per Area'] = [dat_Op['Site Horizontal Infrared Radiation Rate per Area'].iloc[hourOfYear]]
        temp_dict_ZoneDyn['Site Diffuse Solar Radiation Rate per Area'] = [dat_Op['Site Diffuse Solar Radiation Rate per Area'].iloc[hourOfYear]]
        temp_dict_ZoneDyn['Site Direct Solar Radiation Rate per Area'] = [dat_Op['Site Direct Solar Radiation Rate per Area'].iloc[hourOfYear]]
        temp_dict_ZoneDyn['Electric Equipment Electricity Rate'] = [dat_Op['Electric Equipment Electricity Rate'].iloc[hourOfYear]]
        temp_dict_ZoneDyn['People Occupant Count'] = [dat_Op['People Occupant Count'].iloc[hourOfYear]]
        temp_DF_ZoneDyn = pd.DataFrame(temp_dict_ZoneDyn)

        ZoneDynamicsModel_obj.add_operational_data(temp_DF_ZoneDyn,ZMAT_DF)
        ## update control-oriented models if necessary

        # Just a placeholder

        tim=tim+3600
    model.terminate()