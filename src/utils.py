# --------------------------------------------------
#  Paths, Global variables, and other utilities
#  Barbosa, J. (2024)
#  mail@juliabarbosa.net
# -------------------------------------------------- 

import os
from enum import Enum

# -- Global variables --
INPUT_DIR = "input" 
QP_MODEL_DIR = "QP"
MCP_MODEL_DIR= "MCP"
RUNS_DIR = "runs"
MAPS_DIR = "maps"

class ModelType(Enum):
   QP = "qp"
   MCP = "mcp"

