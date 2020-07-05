from caesar.loader import load
from caesar.main import CAESAR
from caesar.driver import drive
#from caesar.group_funcs import get_periodic_r

from caesar.old_loader import load as old_load

def quick_load(*args, **kwargs):
    import warnings
    warnings.warn('The quick-loader is now the default behavior. The -q and --quick flags will be removed soon.', stacklevel=2)
    return load(*args, **kwargs)

