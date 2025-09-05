"""Global configuration settings for the pipeline"""
import warnings
from openmm.unit import kelvin, bar, nanoseconds, femtoseconds, picosecond
from submit import MARTINI

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global settings
INPDB = 'kras.pdb'

# Production parameters
TEMPERATURE = 300*kelvin
PRESSURE = 1*bar
TOTAL_TIME = 1000*nanoseconds 
TSTEP = 20*femtoseconds if MARTINI else 2*femtoseconds
GAMMA = 5/picosecond if MARTINI else 1/picosecond  # 5 for CG, 1 for AA
NOUT = 1000 if MARTINI else 10000  # 1000 for CG, 10000 for AA

