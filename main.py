# -----------------------------------------------------------------------------
#  Example of a main script to run the model, with the necessary instructions
#  
#  - Requires a GAMS license including the PATH MCP Solver (MCP Model), alternatively you 
#  can edit the MCP/model.gms file to use IPOPT or another solver non-linear solver.  
#  - Requires the GAMS Python API to be installed.
#
#  Barbosa, J. (2024)
#  mail@juliabarbosa.net
# ----------------------------------------------------------------------------

from src.utils import *
from src.gams_input import *

import shutil

if __name__ == "__main__":

   # -- Parameters -- #    
   mapname = "base" # .xlsx file with the data of the model you want to run located at MAPS_DIR
   scenario = "summer"
   # ModelType.QP -> Social Welfare Maximization ; ModelType.MCP -> Cournot Competition Model
   model_type = ModelType.QP
   agent_multiplier = 1 # Agent multiplier for the MCP model 
   results_file = "results.gdx" # File to save the results of the model

   # -- Start Simulations -- #
   print(f'\n## -- Starting simulations for map {mapname} -- ## \n')
   fname = os.path.join(MAPS_DIR, mapname+".xlsx")

   # -- Create Results Dir at RUNS_DIR and copy the Input Map for later reference -- #
   print(f'Creating Results Dir for: {fname} - scenario: {scenario}')
   sim_id = scenario
   results_dir = os.path.join(RUNS_DIR,mapname,  sim_id)
   os.makedirs(results_dir, exist_ok=True)
   shutil.copy(fname, results_dir)

   # -- Run the Model -- #
   print(f'Running Model for {fname} - scenario: {scenario}')
   t = time.time()

   if model_type == ModelType.QP:
      set_up_model(fname, ModelType.QP, am = agent_multiplier, scenario=scenario)
      run_gams_model(get_qp_dir(), solutions_file = os.path.join("..", results_dir, results_file))
   
   elif model_type == ModelType.MCP:
      set_up_model(fname, ModelType.MCP, am = agent_multiplier, scenario=scenario)
      run_gams_model(get_mcp_dir(), solutions_file = os.path.join("..", results_dir, results_file))   
   print(f"   Time elapsed: {time.time() - t}")

print("## -- All Simulations Completed -- ##")