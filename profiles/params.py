from lmfit import Parameter
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