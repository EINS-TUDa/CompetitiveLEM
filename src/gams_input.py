# --------------------------------------------------
#  GAMS input file generator
#  Barbosa, J. (2024)
#  mail@juliabarbosa.net
# -------------------------------------------------- 

import os
import  pandas as pd
import numpy as np

import gams.transfer as gt
from gams import GamsWorkspace


import time

from src.utils import QP_MODEL_DIR, MCP_MODEL_DIR, ModelType, RUNS_DIR, MAPS_DIR

# -- Helpers --
def _get_model_path(fname, ty:ModelType):
   """Returns the path to the model file"""
   if ty == ModelType.QP:
      return os.path.join(QP_MODEL_DIR, fname)
   elif ty == ModelType.MCP:
      return os.path.join(MCP_MODEL_DIR, fname)
   else:
      raise ValueError("Invalid model type")

def get_mcp_dir(fname=''):
   return _get_model_path(fname, ModelType.MCP)

def get_qp_dir(fname=''):
   return _get_model_path(fname, ModelType.QP)


class GAMSModel:
   """
   Abstract GAMS Model class.
   Adapted from modelInterface.py @ git@github.com:JP-Barbosa/gdxpy.git 
   It is used to create the GAMS model and write it to a .gdx file.
   """
   
   def __init__(self) -> None:
      self.m = gt.Container()

      self.d_sets = {} #1D sets
      self.dd_sets = {} #2+D sets
      self.param_sets = {}
      self.params = {}

      self.sets_gt = []
      self.params_gt = []
      
      self.params_records: dict[str, pd.DataFrame]= dict()
      self.sets_records: dict[str,list[any]] = dict()

   def sets2container(self):
      """Add the sets to the .gdx data container"""
      cont = 0
      for name, description in self.d_sets.items():
         # Check if already in container, aka in sets
         if name not in [s.name for s in self.sets_gt]:
            cont += 1
            self.sets_gt.append(gt.Set(self.m, name, description=description))
      for name, infos in self.dd_sets.items():
         if name not in [s.name for s in self.sets_gt]:
            cont += 1
            #Todo: solve hard codding here!
            self.sets_gt.append(gt.Set(self.m, name, domain=infos[1:], description=infos[0], records=None))
      for name, records in self.param_sets.items():
         if name not in [s.name for s in self.sets_gt]:
            cont += 1
            self.sets_gt.append(gt.Set(self.m, name, records=records, description=""))

   def params2container(self):
      """Add the parameters to the .gdx data container"""
      cont = 0
      for name, domain in self.params.items():
         if name not in [s.name for s in self.sets_gt]:
            cont += 1
            self.params_gt.append(gt.Parameter(self.m, name, domain=domain))
   
   def set_records(self, ty:str, name:str, records):
      """Set the records for a set or parameter"""
      if records is not None and len(records) == 0:
         raise ValueError("Records cannot be empty")
      # valid types
      assert ty in ["set", "param"]
      symbols = getattr(self, f"{ty}s_gt")

      for s in symbols:
         if s.name == name:
            s.setRecords(records)
            break
      else:
         raise ValueError(f"Symbol {name} not found in {ty}s_gt")

   def records2container(self, ty:str):
         """Transfer the records to the .gdx container. Note that the sets and parameters must be already in the container.
         Call this method only after set2container and params2container have been called.
         """
         # verify ty is valid
         assert ty in ["set", "param"]
         symbols = getattr(self, f"{ty}s_records")
         
         try:
            for sname in symbols.keys():
               self.get_symbol(sname).setRecords(symbols[sname])
         except Exception as e:
            print(f"Error setting records for {ty} {sname}")
            raise e

   def write(self, fname):
      """Write the model to a .gdx file. The sets and parameters, and their records must be already in the container."""
      for pname in self.params_records.keys():
         try:
            self.get_symbol(pname).setRecords(self.params_records[pname])
            s = self.get_symbol(pname).isValid(verbose=True)
         except Exception as e:
            print(f"Error setting records for parameter {pname}")
            raise e
         
      # Check if all symbols are valid
      for s in self.sets_gt + self.params_gt:
         try:
            s.isValid(verbose=True)
         except TypeError as e:
               print(f"Error checking symbol {s.name}... Now removing it from container")
               # remove s from container
               self.m.removeSymbols(s.name)
   
      self.m.write(fname)

   def get_symbol(self, sname):
      """Return the symbol with name sname. Raises a ValueError if not found."""
      for s in self.sets_gt + self.params_gt:
         if s.name == sname:
            return s
      else:
         raise ValueError(f"Symbol {sname} not found!")

   def get_records(self, sname):
      """Return the records of the symbol with name sname. Raises a ValueError if not found."""
      for s in self.sets_gt:
         if s.name == sname:
            dim = len(s.domain)
            # return dim columns
            return [np.array(s.records.iloc[:, x].values) for x in range(dim)]
      for s in self.params_gt:
         if s.name == sname:
            return s.records
      else:
         raise ValueError(f"Symbol {sname} not found!")

   def _read_input_file(fname):
      """Reads the input file and returns a dictionary with the sets and parameters"""
      with open(fname, "r") as f:
         lines = f.readlines()
         lines = [l.strip() for l in lines if l.strip() != ""]
         lines = [l for l in lines if l[0] != "#"]
         
      entries = " ".join(lines).split(";")

      pass

   def add_set_record(self, name:str, values:list[any]):
      """Add a set record to the model"""
      self.sets_records[name] = values
      pass

   def add_param_record(self, name:str, values:list[any]):
      """Add a parameter record to the model"""
      if name in self.params_records.keys():
         self.params_records[name] = pd.concat([self.params_records[name], values])
      else:
         self.params_records[name] = values
      pass

class MyModel(GAMSModel):
   """Model class for the Concrete Local Energy Market Model."""
   def __init__(self) -> None:
      super().__init__()

      # Sets
      self.d_sets = {"a": "Agents",
                     "c": "Commodities",
                     "t": "Timesteps",
                     }
      
      self.dd_sets = {"p": ["Conversion Processes","a","c","c"]
                      }
      
      self.param_sets = {"ParamSetSocial": ["D0", 
                                            "elasticity",
                                            ],
                        "ParamSetAgentCommodity":["maxGen",
                                                   "strCap",
                                                   "strEff",
                                                   "bindedGenFactor",
                                                   "costGen",
                                                   "availbilityGiven",
                                                   "hasStorage",
                                                   "storagePower",
                                                   "wasteCost"
                                                   ],
                        "ParamSetAgentCommodityTime":["minLoad",
                                                       "maxLoad", 
                                                       "availability",
                                                       ],
                        "ParamSetProcess": ["pEff",
                                            "pmaxPower",
                                            ], 
                        "ParamSetSLR": ["hasSLR", 
                                        "maxRegPrice",
                                       ],
                         }

      # Parameters
      self.params = {"ParamSocial": ["c", "t", "ParamSetSocial"],
                     "ParamAC": ["a", "c", "ParamSetAgentCommodity"],
                     "ParamACT": ["a", "c", "t", "ParamSetAgentCommodityTime"],
                     "ParamAP" :["a","c","c","ParamSetProces"],
                     "ParamSLR": ["c", "ParamSetSLR"],
                     "validAC": ["a", "c"],
                     "validWaste": ["a", "c"],
                     }
      
      # Records
      self.social_records = None
      self.agent_commodity_records = None
      self.agent_commodity_time_records = None
   
   def add_param_record(self, name: str, values: pd.DataFrame):
      """ Overwrite the add_param_record method to handle the case where the parameter is not in the model or in the param_sets."""
      if values is None or values.shape[0] == 0:
         print(f"Empty dataframe for parameter {name}! Skipping...")
         return
     
      name_found = False
      if name not in self.params.keys():
         for paramSet, paramNames in self.param_sets.items():
            if name_found:
               break
            if name in paramNames:
               for pname, pvalues in self.params.items():
                  if paramSet in pvalues:
                     ii = pvalues.index(paramSet)
                     # insert a column in the values dataframe where every line has the pnam
                     values.insert(ii, "", [name for i in range(values.shape[0])])
                     name_found = True
                     name = pname
                     break
      
         if not name_found:
            raise ValueError(f"Parameter {name} not found!")
                  
      return super().add_param_record(name, values)
         
class InputReader:
   """Class to read the .xlxs input file (DataMap) and return the sets and parameters for the GAMS model."""

   def __init__(self, fname:str, agent_multiplier = 1, scenario = None) -> None:
      self.fname = fname


      self.scenarios = pd.read_excel(fname, sheet_name="Scenarios", skiprows=[0], header=0) 
      social_params_name, time_params_name, self.scenario = self._get_scenario(scenario)

      self.agents = pd.read_excel(fname, sheet_name="Agents", skiprows=[0], header=0)
      self.agents_params = pd.read_excel(fname, sheet_name="AgentParams", skiprows=[0], header=0)
      self.social_params = pd.read_excel(fname, sheet_name=social_params_name,skiprows=[0], header=0)
      self.time_params = pd.read_excel(fname, sheet_name=time_params_name, skiprows=[0], header=0)
      self.process_params = pd.read_excel(fname, sheet_name='ProcessParams', skiprows=[0], header=0)
      
      self.agent_multiplier = agent_multiplier

      self._generate_agents_sets()

   def _get_scenario(self, scenario=None):
      # If scenario is None return the first scenario
      if scenario is None:
         row = self.scenarios.iloc[0, :]
         
      else: 
         try:
            row = self.scenarios.loc[self.scenarios.loc[:,"Scenario"] == scenario,:].iloc[0,:]
         except KeyError:
            raise ValueError(f"Scenario {scenario} not found!")
      return row["SocialParams"], row["TimeParams"], row["Scenario"]

   def get_commodities(self):
      return self.agents_params["Commodity"].unique()

   def get_processes(self):
      col_names = ["a","ci", "co"]
      p = []
      for row in self.process_params.itertuples():
         p.append([getattr(row, name) for name in col_names])
      # replace by n of agents with correct name
      p = pd.DataFrame(p, columns=col_names)
      p = self._perform_agent_copy(self.agents, p, agent_col="a")
      return p

   def get_agents(self):
      # concatenate all entries on the "a" column
      return self.agents["a"].sum()

   def _get_n_agents_of_type(self, agent_type):
      return self.agents[self.agents["Agent"] == agent_type]["N"].sum()

   def get_timesteps(self):
      return self.time_params["Time"].unique()

   def _generate_agents_sets(self):
      # apply multiplication factor
      # Take agents that multiply is not zero and multiply the multiply column by the agent_multiplier
      f = lambda x: x["N"]*x["Multiply"]* self.agent_multiplier if x["Multiply"] != 0 else x["N"]
      self.agents['N'] = self.agents.apply(f, axis=1)

      fun = lambda x: self.generate_names(x["Agent"], x["N"])
      self.agents["a"] =  self.agents.apply(fun, axis=1)

   def get_time_params(self):


      t_param = self._read_time_table(self.time_params) 
      t_param = pd.DataFrame(t_param, columns=["Agent", "Commodity", "Param", "Time", "Value"])
      # reorder the columns to match the order of the sets
      t_param = t_param[["Agent", "Commodity", "Time", "Param", "Value"]]
      t_param = self._perform_agent_copy(self.agents, t_param)
      
      return t_param

   def _divide_by_n_agents(self, df:pd.DataFrame, params_to_divide:list[str], agent_col:str = "Agent"):
      "Divide the values of the parameters in params_to_divide by the number of agents of that type"
      
      # certify that the columns are in the dataframe
      for p in params_to_divide:
         assert p in df.columns, f"Column {p} not found in dataframe!"


      for a in df[agent_col].unique():
         n = self._get_n_agents_of_type(a)
         df.loc[df[agent_col] == a, params_to_divide] = df.loc[df[agent_col] == a, params_to_divide]/n
      return df

   def get_valid_agent_commodities(self):
      "read the Agent Params sheet and return the valid agent-commodity pairs"
      df = self.agents_params.loc[:,["Agent","Commodity"]]
      df["Value"] = 1
      df = self._perform_agent_copy(self.agents, df)
      return df
   
   def get_valid_waste(self):
      "read the Agent Params and check which agents have wasteCost != -1"
      df = self.agents_params.loc[:,["Agent","Commodity","wasteCost"]]
      df = df[df["wasteCost"] != -1]
      df = df[["Agent","Commodity"]]
      df["Value"] = 1
      df = self._perform_agent_copy(self.agents, df)
      if len(df) == 0:
         return None
      return df

   @staticmethod
   def _read_time_table(table:pd.DataFrame, ref_col = "Time"):
      ret = []
      for name, col in table.items():
         if name != ref_col:
            # remove spaces
            #aa,cc,param 
            vals = name.replace(" ", "").split(",")
            for i in range(len(col)):
               ret.append((*vals, ts[i],  col[i]))
         else:
            ts = col
      return ret

   def get_social_params(self):

      data = self._read_time_table(self.social_params)
      data = pd.DataFrame(data, columns=[ "Param","Commodity","Time","Value"])
      # swith the columns to match the order of the sets
      data = data[["Commodity", "Time", "Param", "Value"]]
      return data

   def get_agent_params(self):
      # Some parameters must be divided by the number of agents of that type to divide market power
      # This is done by the function _divide_by_n_agents
      params_to_divide = ["maxGen", "strCap", "strPower"]
      table = self._divide_by_n_agents(self.agents_params, params_to_divide)
         
      df = self._perform_agent_copy(self.agents, table)
      val_vars = [x for x in df.columns if x not in ["Agent", "Commodity"]]
      return pd.melt(df, id_vars = ["Agent", "Commodity"], value_vars = val_vars, var_name="Param", value_name="Value")
   
   def get_process_params(self):
      params_to_divide = ["pmaxPower"]
      table = self._divide_by_n_agents(self.process_params, params_to_divide, agent_col="a")

      val_vars = [x for x in table.columns if x not in ["a", "ci", "co"]]
      params = pd.melt(table, id_vars = ["a", "ci", "co"], value_vars = val_vars, var_name="Param", value_name="Value")
      params = self._perform_agent_copy(self.agents, params, agent_col="a")
      return params

   def get_reg_params(self):
      "Read the reg params table and return a dataframe with the values"
      df = self.scenarios
      # select the colung with SLR on the name 
      reg = [x for x in df.columns if "SLR" in x]
      df = df[["Scenario"] +reg]

      df = df.melt(id_vars = "Scenario", value_vars = reg, var_name="c", value_name="maxRegPrice")

      df["hasSLR"] = df["maxRegPrice"].apply(lambda x : 1 if x != -1 else 0)
      df["c"] = df["c"].apply(lambda x: x.replace("SLR_", ""))

      df = df[df["Scenario"] == self.scenario][["c", "hasSLR", "maxRegPrice"]]
      df = df.melt(id_vars = ["c"], value_vars = ["maxRegPrice", "hasSLR"], var_name="Param", value_name="Value")
      return df  
      

   @staticmethod
   def generate_names(name, n:int):
      return [f"{name}_{i}" for i in range(n)]
   
   @staticmethod
   def _perform_agent_copy(agents: pd.DataFrame, params:pd.DataFrame, agent_col = "Agent"):
     
      ii = params.columns.get_loc(agent_col)
      iii = params.columns == agent_col
     
      for row in params.itertuples():
         lookup_val = row[ii+1]
         row_vals = np.array(row[1:], dtype = 'object')[~iii]
     
         if lookup_val in agents["Agent"].values:
     
            for ai in agents[agents["Agent"] == lookup_val]["a"].values[0]:
               # insert ai at possition ii and dislocate the rest of row values
               rr = np.insert(row_vals, ii, ai)
               new_row = pd.Series(rr, index=params.columns)
               params = pd.concat([params, new_row.to_frame().T],ignore_index=True)
     
            params = params[params[agent_col] != lookup_val]
     
      return params

## -- Main functions -- ##
def set_up_model(excel_input_fname,ty:ModelType, gams_input_fanme="qp_input.gdx", am=1, scenario=None):
   """
   Set up the model and write it to a .gdx file. 
   The model is written to the directory specified by the get_qp_dir or get_mcp_dir functions.

   Parameters:
      excel_input_fname: str
         The name of the excel input file.
      ty: ModelType. ModelType.QP or ModelType.MCP
         The type of model to set up.
      gams_input_fanme: str
         The name of the .gdx file to write the model to.
      am: int
         The agent multiplier. Default is 1. (Meanifully only for MCP. Divides the declared capacity of the type amoung n agents of that type)
      scenario: str
         The scenario to run. Default is None, which runs the first scenario in the Scenarios sheet.
   """

   if ty == ModelType.QP:
      gams_input_fanme = "qp_input.gdx"
      dir_fun = get_qp_dir
   elif ty == ModelType.MCP:
      gams_input_fanme = "mcp_input.gdx"
      dir_fun = get_mcp_dir

   reader = InputReader(excel_input_fname, agent_multiplier=am, scenario=scenario)

   model = MyModel()
   
   model.sets2container()
   model.params2container()

   model.add_set_record("a", reader.get_agents())
   model.add_set_record("t", reader.get_timesteps())
   model.add_set_record("c", reader.get_commodities())
   model.add_set_record("p", reader.get_processes())
   
   model.records2container("set")

   # Add records to parameters
   model.add_param_record("ParamSocial", reader.get_social_params())
   model.add_param_record("ParamAC", reader.get_agent_params())
   model.add_param_record("ParamACT", reader.get_time_params())
   model.add_param_record("ParamAP", reader.get_process_params())
   model.add_param_record("validAC", reader.get_valid_agent_commodities())
   model.add_param_record("validWaste", reader.get_valid_waste())
   model.add_param_record("ParamSLR", reader.get_reg_params())


   model.write(dir_fun(gams_input_fanme))

def run_gams_model(workspace, model_file='model.gms', solutions_file='solutions.gdx'):
   """
   Run the GAMS model and export the solutions to a .gdx file.
   
   Parameters:
      workspace: str
         The directory where the model.gms file is located.
      model_file: str
         The name of the model.gms file. Default is 'model.gms'.
      solutions_file: str
         The name of the .gdx file to write the solutions to. Default is 'solutions.gdx'.
   """
   ws = GamsWorkspace(working_directory=workspace)
   job = ws.add_job_from_file(model_file)
   job.run()
   job.out_db.export(solutions_file)

