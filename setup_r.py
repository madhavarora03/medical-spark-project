from rpy2.robjects.packages import importr

base = importr("base", lib_loc="R")
utils = importr("utils", lib_loc="R")
import rpy2.robjects.packages as rpackages

forest = rpackages.importr("randomForestSRC", lib_loc="R")
from rpy2 import robjects as ro

R = ro.r

from rpy2.robjects import pandas2ri

pandas2ri.activate()