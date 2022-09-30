# EnergyPlus, Copyright (c) 1996-2021, The Board of Trustees of the University
# of Illinois, The Regents of the University of California, through Lawrence
# Berkeley National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy), Oak Ridge National Laboratory, managed by UT-
# Battelle, Alliance for Sustainable Energy, LLC, and other contributors. All
# rights reserved.
#
# NOTICE: This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
# Software to reproduce, distribute copies to the public, prepare derivative
# works, and perform publicly and display publicly, and to permit others to do
# so.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# (1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
# (2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# (3) Neither the name of the University of California, Lawrence Berkeley
#     National Laboratory, the University of Illinois, U.S. Dept. of Energy nor
#     the names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# (4) Use of EnergyPlus(TM) Name. If Licensee (i) distributes the software in
#     stand-alone form without changes from the version obtained under this
#     License, or (ii) Licensee makes a reference solely to the software
#     portion of its product, Licensee must refer to the software as
#     "EnergyPlus version X" software, where "X" is the version number Licensee
#     obtained under this License and may not use a different name for the
#     software. Except as specifically required in this Section (4), Licensee
#     shall not use in a company name, a product name, in advertising,
#     publicity, or other promotional activities any name, trade name,
#     trademark, logo, or other designation of "EnergyPlus", "E+", "e+" or
#     confusingly similar designation, without the U.S. Department of Energy's
#     prior written consent.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
import sys
import random

print("asfaf")
print(sys.path)

from pyenergyplus.plugin import EnergyPlusPlugin
def read_BSDF():

    with open(r'/scratch/user/ffeng/MPC/Case1/BSDF_Shade.txt','r') as fp:
        count = 0
        for line in fp.readlines():
            if count == 0:
                controlVariable = [float(s) for s in line.split(',')]
            elif count == 1:
                TfSol =[float(s) for s in line.split(',')]
            elif count == 2:
                RbSol =[float(s) for s in line.split(',')]
            elif count == 3:
                TfVis =[float(s) for s in line.split(',')]
            elif count == 4:
                RbVis =[float(s) for s in line.split(',')]
            elif count == 5:
                fAbs_layer1 =[float(s) for s in line.split(',')]
            elif count == 6:
                bAbs_layer1 =[float(s) for s in line.split(',')]
            elif count == 7:
                fAbs_layer2 =[float(s) for s in line.split(',')]
            else:
                bAbs_layer2 =[float(s) for s in line.split(',')]
            count += 1
        if count <= 2:
            TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2 = 0,0,0,0,0,0,0,0
    return TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2,controlVariable

def read_BSDF_clear():

    with open(r'/scratch/user/ffeng/MPC/Case1/BSDF_clear.txt','r') as fp:
        count = 0
        for line in fp.readlines():
            if count == 0:
                controlVariable = [float(s) for s in line.split(',')]
            elif count == 1:
                TfSol_clear =[float(s) for s in line.split(',')]
            elif count == 2:
                RbSol_clear =[float(s) for s in line.split(',')]
            elif count == 3:
                TfVis_clear =[float(s) for s in line.split(',')]
            elif count == 4:
                RbVis_clear =[float(s) for s in line.split(',')]
            elif count == 5:
                fAbs_layer1_clear =[float(s) for s in line.split(',')]
            elif count == 6:
                bAbs_layer1_clear =[float(s) for s in line.split(',')]
            count += 1
        if count <= 1:
            TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear = 0,0,0,0,0,0
    return TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear,controlVariable


class SetCFSState(EnergyPlusPlugin):
    ShadeStatusInteriorBlindOn = 6
    ShadeStatusOff = 0

    def __init__(self):
        super().__init__()
        self.handles_set = False
        self.handles_set2 = False   # THis is handle_set for VAV and lighting
        self.handle_TestVariable_status = None  # only because actuators can be output vars ... yet
        self.CFS_Glz_Win1 = None
        self.CFS_Glz_Win2 = None
        self.CFS_Glz_Win3 = None
        self.CFS_Glz_Win4 = None
        self.CFS_Glz_Win5 = None
        self.CFS_Glz_Win6 = None
        self.CFS_Glz_Win7 = None
        self.CFS_Glz_Win8 = None
        self.CFS_Glz_Win9 = None
        self.CFS_Glz_Win10 = None
        self.CFS_Glz_Win1_clear = None
        self.CFS_Glz_Win2_clear = None
        self.CFS_Glz_Win3_clear = None
        self.CFS_Glz_Win4_clear = None
        self.CFS_Glz_Win5_clear = None
        self.CFS_Glz_Win6_clear = None
        self.CFS_Glz_Win7_clear = None
        self.CFS_Glz_Win8_clear = None
        self.CFS_Glz_Win9_clear = None
        self.CFS_Glz_Win10_clear = None

        self.Win1_Construct_handle = None
        self.Win2_Construct_handle = None
        self.Win2_Construct_handle = None
        self.Win3_Construct_handle = None
        self.Win4_Construct_handle = None
        self.Win5_Construct_handle = None
        self.Win6_Construct_handle = None
        self.Win7_Construct_handle = None
        self.Win8_Construct_handle = None
        self.Win9_Construct_handle = None
        self.Win10_Construct_handle = None


        self.light_level_handle = None  # only because actuators can be output vars ... yet  
        self.mass_flowrate_handle =  None
        self.SAT_handle = None
    
    def set_matrix_value(self,state,WinNum,TfSol_temp,RbSol_temp,TfVis_temp,RbVis_temp,fAbs_layer1_temp,bAbs_layer1_temp,fAbs_layer2_temp,bAbs_layer2_temp):
        
        #CFS_Glz_WinX_TfSol
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_TfSol".format(WinNum),145,145,TfSol_temp)

        #CFS_Glz_WinX_RbSol
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_RbSol".format(WinNum),145,145,RbSol_temp)

        #CFS_Glz_WinX_Tfvis
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Tfvis".format(WinNum),145,145,TfVis_temp)

        #CFS_Glz_WinX_Rbvis
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Rbvis".format(WinNum),145,145,RbVis_temp)

        #CFS_Glz_WinX_Layer_1_fAbs
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Layer_1_fAbs".format(WinNum),1,145,fAbs_layer1_temp)

        #CFS_Glz_WinX_Layer_1_bAbs
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Layer_1_bAbs".format(WinNum),1,145,bAbs_layer1_temp)

        #CFS_Glz_WinX_Layer_2_bAbs
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Layer_2_fAbs".format(WinNum),1,145,fAbs_layer2_temp)

        #CFS_Glz_WinX_Layer_2_fAbs
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Layer_2_bAbs".format(WinNum),1,145,bAbs_layer2_temp)

        return 0

    def set_matrix_value_clear(self,state,WinNum,TfSol_temp,RbSol_temp,TfVis_temp,RbVis_temp,fAbs_layer1_temp,bAbs_layer1_temp):
        #CFS_Glz_WinX_TfSol
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_TfSol".format(WinNum),145,145,TfSol_temp)

        #CFS_Glz_WinX_RbSol
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_RbSol".format(WinNum),145,145,RbSol_temp)

        #CFS_Glz_WinX_Tfvis
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Tfvis".format(WinNum),145,145,TfVis_temp)

        #CFS_Glz_WinX_Rbvis
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Rbvis".format(WinNum),145,145,RbVis_temp)

        #CFS_Glz_WinX_Layer_1_fAbs
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Layer_1_fAbs".format(WinNum),1,145,fAbs_layer1_temp)

        #CFS_Glz_WinX_Layer_1_bAbs
        self.api.exchange.set_Plugin_Matrix(state, "CFS_Glz_Win{}_Layer_1_bAbs".format(WinNum),1,145,bAbs_layer1_temp)

        return 0 

    def on_begin_zone_timestep_before_init_heat_balance(self, state):  
        if not self.handles_set:    
            ###### Set construction using construction actuator
            ## 1) Get construction handle
            self.CFS_Glz_Win1 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win1")
            self.CFS_Glz_Win2 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win2")
            self.CFS_Glz_Win3 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win3")
            self.CFS_Glz_Win4 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win4")
            self.CFS_Glz_Win5 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win5")
            self.CFS_Glz_Win6 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win6")
            self.CFS_Glz_Win7 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win7")
            self.CFS_Glz_Win8 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win8")
            self.CFS_Glz_Win9 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win9")
            self.CFS_Glz_Win10 = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win10")
            self.CFS_Glz_Win1_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win1_clear")
            self.CFS_Glz_Win2_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win2_clear")
            self.CFS_Glz_Win3_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win3_clear")
            self.CFS_Glz_Win4_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win4_clear")
            self.CFS_Glz_Win5_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win5_clear")
            self.CFS_Glz_Win6_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win6_clear")
            self.CFS_Glz_Win7_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win7_clear")
            self.CFS_Glz_Win8_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win8_clear")
            self.CFS_Glz_Win9_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win9_clear")
            self.CFS_Glz_Win10_clear = self.api.exchange.get_construction_handle(state, "CFS_Glz_Win10_clear")
            
            ## 2) Construction actuator
            self.Win1_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win1")
            self.Win2_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win2")
            self.Win3_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win3")
            self.Win4_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win4")
            self.Win5_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win5")
            self.Win6_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win6")  
            self.Win7_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win7")
            self.Win8_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win8")
            self.Win9_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win9")
            self.Win10_Construct_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Surface",
                                                                                "Construction State",
                                                                                "28908_Wall_9_0_0_0_0_0_Win10")  
            self.handles_set = True 
        
        ## This part implements the control strategy
        ## Step 0: print the currrent DayOfSim and TimeStepOfHour
        print("Day of Year: ", self.api.exchange.day_of_year(state),"   Hour:", self.api.exchange.hour(state))


        if (self.api.exchange.current_environment_num(state)<3):  ## Ony apply this for run period.
            return 0
        else:
            #return 0 # for debug
            print(self.api.exchange.current_environment_num(state))

            ## Step 1: Read control signal from local log file.
                       
            TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2 ,controlVariable=  read_BSDF()
            TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear,controlVariable =  read_BSDF_clear()
            
            if not controlVariable[-1]:
                # If the control on-off signal is False, turn off control
                print("Turn off MPC controller")
                return 0
            print("read completed")
            blind_height = controlVariable[2]      # random.randint(0,10)   # random integer between [0,11)

            print("Bling Angle:{},Blind Height is {}".format(controlVariable[1],controlVariable[2]))
            
            if blind_height >= 1:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win1_Construct_handle, self.CFS_Glz_Win1)
            
                self.set_matrix_value(state,1,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win1_Construct_handle, self.CFS_Glz_Win1_clear)
                self.set_matrix_value_clear(state,1,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 2
            if blind_height >= 2:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win2_Construct_handle, self.CFS_Glz_Win2)
                self.set_matrix_value(state,2,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win2_Construct_handle, self.CFS_Glz_Win2_clear)
                self.set_matrix_value_clear(state,2,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)

            ## Window 3
            if blind_height >= 3:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win3_Construct_handle, self.CFS_Glz_Win3)
                self.set_matrix_value(state,3,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win3_Construct_handle, self.CFS_Glz_Win3_clear)
                self.set_matrix_value_clear(state,3,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 4
            if blind_height >= 4:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win4_Construct_handle, self.CFS_Glz_Win4)
                self.set_matrix_value(state,4,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win4_Construct_handle, self.CFS_Glz_Win4_clear)
                self.set_matrix_value_clear(state,4,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 5
            if blind_height >= 5:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win5_Construct_handle, self.CFS_Glz_Win5)
                self.set_matrix_value(state,5,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win5_Construct_handle, self.CFS_Glz_Win5_clear)
                self.set_matrix_value_clear(state,5,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 6
            if blind_height >= 6:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win6_Construct_handle, self.CFS_Glz_Win6)
                self.set_matrix_value(state,6,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win6_Construct_handle, self.CFS_Glz_Win6_clear)
                self.set_matrix_value_clear(state,6,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 7
            if blind_height >= 7:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win7_Construct_handle, self.CFS_Glz_Win7)
                self.set_matrix_value(state,7,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win7_Construct_handle, self.CFS_Glz_Win7_clear)
                self.set_matrix_value_clear(state,7,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 8
            if blind_height >= 8:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win8_Construct_handle, self.CFS_Glz_Win8)
                self.set_matrix_value(state,8,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win8_Construct_handle, self.CFS_Glz_Win8_clear)
                self.set_matrix_value_clear(state,8,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 9
            if blind_height >= 9:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win9_Construct_handle, self.CFS_Glz_Win9)
                self.set_matrix_value(state,9,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win9_Construct_handle, self.CFS_Glz_Win9_clear)
                self.set_matrix_value_clear(state,9,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            ## Window 10
            if blind_height >= 10:
                # Turn the shade of window 1 on
                self.api.exchange.set_actuator_value(state, self.Win10_Construct_handle, self.CFS_Glz_Win10)
                self.set_matrix_value(state,10,TfSol,RbSol,TfVis,RbVis,fAbs_layer1,bAbs_layer1,fAbs_layer2,bAbs_layer2)
            else:
                # Turn the shade of window 1 off
                self.api.exchange.set_actuator_value(state, self.Win10_Construct_handle, self.CFS_Glz_Win10_clear)
                self.set_matrix_value_clear(state,10,TfSol_clear,RbSol_clear,TfVis_clear,RbVis_clear,fAbs_layer1_clear,bAbs_layer1_clear)
            
            print("CFS control deployed")
        return 0    

    def on_begin_zone_timestep_after_init_heat_balance(self, state):    
        # api is ready to execute
        if self.api.exchange.api_data_fully_ready(state):
            # get variable handles if needed
            if not self.handles_set2:
                self.light_level_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "Lights",
                                                                                "Electricity Rate",
                                                                                "light_28908")
                
                self.mass_flowrate_handle = self.api.exchange.get_actuator_handle(state,
                                                                                "AirTerminal:SingleDuct:ConstantVolume:NoReheat",
                                                                                "Mass Flow Rate",
                                                                                "Air Terminal Single Duct CAV 28908")
                self.SAT_handle   = self.api.exchange.get_actuator_handle(state, 'Schedule:Compact', 'Schedule Value', 'SAT_Setpoint_EMS')
            # self.SAT_handle = self.api.exchange.get_actuator_handle(state,
            # )
                self.handles_set2 = True


            _,_,_,_,_,_,_,_,controlVariable =  read_BSDF()
            Light_lvl = controlVariable[3] #random.random()*400
            print(Light_lvl)
            self.api.exchange.set_actuator_value(state, self.light_level_handle, Light_lvl)

            Mass_FR = controlVariable[4] #random.random()*0.25 +0.10
            print(self.mass_flowrate_handle)
            self.api.exchange.set_actuator_value(state, self.mass_flowrate_handle, Mass_FR)   # This is implemented using EMS

            SAT_value = controlVariable[5] #random.random()*6 +12   # in heating case. 
            self.api.exchange.set_actuator_value(state, self.SAT_handle, SAT_value)
            
            print("Lighting and HVAC system control deployed")
            return 0
        else:
            return 0

