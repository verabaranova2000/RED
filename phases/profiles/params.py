from lmfit import Parameter
import re, math
from utils.cif_extract import get_value_for_atom
## ========= Набор параметров ===========


## ==== Коэффициенты при полиномах Лежандра (bckg0 ... ) ====
def create_par_bckg(number_of_terms):
  objects = {}
  for i in range(number_of_terms):
    object_name = 'bckg' + str(i)
    objects[object_name] = Parameter(object_name)

    objects.get(object_name).value = 0
    objects.get(object_name)._vary = False
  return objects