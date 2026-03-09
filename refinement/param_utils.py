from lmfit import Parameters, Parameter 


# ==== Объединение параметров (параметры всех фаз в проекте и фона) с учетом setting ====
def prepare_Parameters(project_object=None, phase_object=None, profile=True, background=True):
    assert project_object==None or phase_object==None
    set_of_Phases=[]                                    # Список, куда поместим все фазы
    if project_object!=None: set_of_Phases=[project_object.__dict__.get('Phase'+str(KPhase)) for KPhase in range(1, project_object.NPhases+1)]
    elif phase_object!=None: set_of_Phases = [phase_object]
    dict_all_pars={}                                    # словарь, куда будем заносить параметры всех фаз

    for Phasei in set_of_Phases:                        # пробегаемся по фазам из набора

      typeref          = Phasei.setting['typeref']
      corrections      = Phasei.setting['corrections']
      calibrate        = Phasei.setting['calibrate']
      calibration_mode = Phasei.setting['calibration mode']

      """"" 1a. (Rietveld). Объединение словарей с парметрами в один словарь """""
      if typeref=='Rietveld':
        for pars_block_ph in [Phasei.param_cell, Phasei.param_scale_shift_phvol_Biso_overall, Phasei.param_profile]:                        # Добавляем параметры текущей фазы Phasei
          if   profile==True:                                          dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))
          elif profile==False and pars_block_ph!=Phasei.param_profile: dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))
        for atom in Phasei.atoms:                                                                                                            # Добавляем параметры атомов в текущей фазе Phasei:
          dict_all_pars=dict(list(dict_all_pars.items()) + list(atom.params.items()))
        if len(corrections)!=0:                                                                                                              # Добавляем интенсивности, которые нужно уточнить
          for h,k,l in corrections:
            I_name=Phasei.prefix + 'I_' + hkl_to_str([h,k,l])
            dict_all_pars[I_name]=Phasei.param_intensity[I_name]

      """"" 1b. (le Beil). Объединение словарей с парметрами в один словарь """""
      if typeref=='le Beil':
        for pars_block_ph in [Phasei.param_cell, Phasei.param_scale_shift_phvol_Biso_overall, Phasei.param_profile, Phasei.param_intensity]:  # Добавляем параметры текущей фазы Phasei
          if   profile==True:                                          dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))
          elif profile==False and pars_block_ph!=Phasei.param_profile: dict_all_pars=dict(list(dict_all_pars.items()) + list(pars_block_ph.items()))


      """"" 2. (calibrate). Объединение словарей с парметрами в один словарь """""
      if   (calibration_mode==True) and (calibrate=='all'): dict_all_pars=dict(list(dict_all_pars.items()) + list(Phasei.param_delta.items()))
      elif (calibration_mode==True) and (calibrate!=[]):
        #delta_for_ref=[Phasei.prefix+'delta_'+'_'.join([str(i) for i in hkl]) for hkl in Phasei.setting['calibrate']]
        delta_for_ref=[f"{Phasei.prefix}delta_{hkl_to_str(hkl)}" for hkl in Phasei.setting['calibrate']]
        dict_all_pars.update({k: Phasei.param_delta[k] for k in delta_for_ref if k in Phasei.param_delta})

    """"" 3. Создание объекта типа Parameters """""
    params_ph=Parameters()
    for k,v in dict_all_pars.items():
      if   profile==True:                            params_ph.add(v)
      elif profile==False and k.split('_')[-1] not in ['phvol','shift']: params_ph.add(v)
    params_ph

    """"" 3. Добавление параметров фона, если ключевое слово background=True """""
    if (project_object!=None) and (background==True) and (profile==True):
      for k_bckg, v_bckg in project_object.Profile_points.params.items():
        params_ph.add(v_bckg)

    """"" 4. Наложение условий на параметры, если количество фаз > 1 """""
    if (profile==True) and (project_object!=None) and (project_object.NPhases>1):
      for KPhase in range(2,project_object.NPhases+1):
        params_ph['Phase'+str(KPhase)+'_scale'].expr  ='Phase1_scale'
        params_ph['Phase'+str(KPhase)+'_shift'].expr  ='Phase1_shift'
        #params_ph['Phase'+str(KPhase)+'_phvol'].value = 0.5
        #params_ph['Phase'+str(KPhase)+'_phvol'].init_value = 0.5
  
    return params_ph



# ====== Глубокая копия набора параметров (объекта типа Parameters) ========
def deepcopy_params(params):
    params_new_dict={}
    for k,v in params.items():             # Пробегаемся по каждому параметру (здесь параметр - это значение в словаре)
      pari={}                              # Создадим словарь. В него будем копировать параметр v
      for k1,v1 in v.__dict__.items():     # Пробегаемся по парам ключ-значение, которые входят в параметр
        if k1!='from_internal':
          pari[k1]=v1               ##### Всё, скопировали информацию о параметре в словарь
      par_new=Parameter(name=pari.get('name'))   # Это - копия параметра. Его нужно наполнить содержанием из словаря pari
      for k2,v2 in pari.items():
         par_new.__dict__[k2]=v2           # Заполним словарь значениями по ключам
      params_new_dict[k]=par_new           # Добавим полученный параметр в словарь параметров

    params_new=Parameters()
    for k,v in params_new_dict.items():
      params_new.add(v)
    return params_new



#@title 3. Подготовка набора параметров к новому уточнению ***(с учетом 'm' в hkl)

def params_for_next(model_result, canсel_lastref=None, undate_init_val='all', fix=True, refonly=None, segment=None):    # fixall=True - зафиксировать все параметры, init_val_undate - заменить init_value на уточненные значения
    out=model_result
    pars_new=deepcopy_params(out.params)                                        # Создаем глубокую копию, чтобы не изменялось состояние объекта!!!

    """"""" 1. Отмена результатов последнего уточнения для перечисленных в квадратных скобках параметров """""""
    if canсel_lastref is not None and type(canсel_lastref)==list:                     # чтобы отменить результаты последнего уточнения некоторых параметров и вернуться к начальным значениям
      for par in canсel_lastref:
        assert par in [k for k,v in pars_new.items()]                           # прерываем, если названия параметра нет в списке
        pars_new.get(par).value=out.init_params.get(par).value                    # открываем параметр для уточнения

    """"""" 2. Обновляем начальные значения (init_value = value) либо для всех параметров, либо для перечисленных в квадратных скобках параметров. Теперь уточненные на предыдущем шаге величины будут отправными точками в следующем уточнении"""""""
    if undate_init_val=='all':                      # Обновляем начальные значения (init_value = value)
      for v,k in pars_new.items():
        k.init_value=k.value
    if undate_init_val!='all' and type(undate_init_val)==list:
      for par in undate_init_val:
        assert par in [k for k,v in pars_new.items()]
        pars_new.get(par).init_value=out.params.get(par).value

    """"""" 3. Обновляем набор параметров I_hkl """""""
    for KPhase in range(1, pr.NPhases+1):             # пробегаемся по фазам в проекте
      Phasei      = pr.__dict__.get('Phase'+str(KPhase))
      corrections = Phasei.setting['corrections']
      if len(corrections)!=0:
        for h,k,l in Phasei.setting['corrections']:
          I_name=f"{Phasei.prefix}I_{hkl_to_str([int(h), int(k), int(l)])}"
          if I_name not in pars_new: pars_new.add(Phasei.param_intensity[I_name])


    """"""" 4. Фиксируем все параметры """""""
    if fix==True:                              # Фиксируем все параметры
      for v,k in pars_new.items():
        k.vary=False

    """"""" 5. Открываем для уточнения параметры, перечисленные в квадратных скобках """""""
    if refonly!=None and type(refonly)==list:
      if 'I_hkl' in refonly:                                                                                  # Если хотим уточнить все интенсивности в методе le Beil
        for par in [k for k,v in pars_new.items() if '_I_' in k]: pars_new.get(par).vary=True                 # Открываем все интенсивности для уточнения

      if 'delta_hkl' in refonly:                                                                                  # Если хотим уточнить сдвиги всех пиков
        for par in [k for k,v in pars_new.items() if '_delta_' in k]: pars_new.get(par).vary=True                 # Открываем все сдвиги пиков для уточнения

      for par in refonly:                                                                                     # Если подаем список параметров для уточнения
        assert (par in [k for k,v in pars_new.items()]+['I_hkl']+['delta_hkl']) or ('_I_hkl' in par) or ('_I_inside' in par) or ('_profile' in par) # прерываем, если названия параметра нет в списке
        if par in [k for k,v in pars_new.items()] and (pars_new.get(par).expr==None):
          pars_new.get(par).vary=True                                                           # открываем параметр для уточнения

        #### Уточнение группы параметров для какой-то фазы
        if '_I_hkl' in par:                                                       # Если хотим уточнить все интенсивности только одной фазы
          prefix_KPhase=par.split('_')[0]                                         # Префикс этой фазы, интенсивности которой хотим уточнить
          for parI in [k for k,v in pars_new.items() if prefix_KPhase+'_I_' in k]: pars_new.get(parI).vary=True

        if '_I_inside' in par:                                                                                              # Если хотим уточнить все интенсивности только одной фазы
          prefix_KPhase=par.split('_')[0]                                                                                   # Префикс этой фазы, интенсивности которой хотим уточнить
          segm = out.userkws['axes'] if segment==None else segment
          my_phase               = pr.__dict__.get(prefix_KPhase.replace('_',''))
          for line in my_phase.bragg_positions:
            hi,ki,li=line[:3]
            if check_hkl_in_segment(segm, my_phase,h=hi,k=ki,l=li):
              I_name=f"{my_phase.prefix}I_{hkl_to_str([int(hi), int(ki), int(li)])}"                                                      # если рефлекс лежит внутри уточняемого диапазона, он нам нужен
              if I_name not in pars_new: pars_new.add(my_phase.param_intensity[I_name])                                     # Добавляем интенсивность рефлекса в набор параметров для уточнения, если её там нет
              pars_new.get(I_name).vary=True                                                                                # Открываем её для уточнения
              if pr.__dict__.get(prefix_KPhase.replace('_','')).setting['typeref']!='le Beil' and [hi,ki,li] not in pr.__dict__.get(prefix_KPhase.replace('_','')).setting['corrections']: pr.__dict__.get(prefix_KPhase.replace('_','')).setting['corrections'].append([hi,ki,li])  # Добавляем индексы этих рефлексов в sitting для фазы

        #### Уточнение группы параметров для какой-то фазы
        if '_delta_hkl' in par:                                                   # Если хотим уточнить сдвиги всех пиков только одной фазы
          prefix_KPhase=par.split('_')[0]                                         # Префикс этой фазы, сдвиги пиков которой хотим уточнить
          for pardelta in [k for k,v in pars_new.items() if prefix_KPhase+'_delta_' in k]: pars_new.get(pardelta).vary=True

        if '_delta_inside' in par:                                                                                          # Если хотим уточнить сдвиги всех пиков только одной фазы
          prefix_KPhase=par.split('_')[0]                                                                                   # Префикс этой фазы, сдвиги пиков которой хотим уточнить
          segm = out.userkws['axes'] if segment==None else segment
          my_phase               = pr.__dict__.get(prefix_KPhase.replace('_',''))
          for line in my_phase.bragg_positions:
            hi,ki,li=line[:3]
            if check_hkl_in_segment(segm, my_phase,h=hi,k=ki,l=li):
              delta_name=f"{my_phase.prefix}delta_{hkl_to_str([int(hi), int(ki), int(li)])}"                                                      # если рефлекс лежит внутри уточняемого диапазона, он нам нужен
              if delta_name not in pars_new: pars_new.add(my_phase.param_delta[delta_name])                                     # Добавляем интенсивность рефлекса в набор параметров для уточнения, если её там нет
              pars_new.get(delta_name).vary=True                                                                                # Открываем её для уточнения
              if [hi,ki,li] not in pr.__dict__.get(prefix_KPhase.replace('_','')).setting['calibrate']: pr.__dict__.get(prefix_KPhase.replace('_','')).setting['calibrate'].append([hi,ki,li])  # Добавляем индексы этих рефлексов в setting для фазы

        if '_profile' in par:                                                   # Если хотим уточнить все параметры формы пиков для какой-то фазы
          prefix_KPhase=par.split('_')[0]
          pars_form = [k for k,v in pars_new.items() if  (len(set(model_list).intersection(k.split('_'))) > 0) and (prefix_KPhase in k)] ### Извлекли названия параметров формы
          for par_form in pars_form:
            if (pars_new.get(par).expr==None): pars_new.get(par_form).vary=True

    ## Фиксация параметров, у которых есть expr
    #for k,v in pars_new.items():
    # if v.expr!=None: pars_new[k].vary=False
    return pars_new




# ====== Постобработка параметров  =====
#         - извлекает значение параметра
#         - считает изменение %
def relative_change(pars, name_of_par):
  val      = pars[name_of_par].value
  init_val = pars[name_of_par].init_value
  delta    = round(float((init_val-val)/init_val*100),3) if init_val!=0 else "∞"
  return f"{name_of_par}={round(val, 6):<12} ({delta:>7} %)"


def val_delta_percent(pars, name_of_par):
  val      = pars[name_of_par].value
  init_val = pars[name_of_par].init_value
  delta    = round(float((init_val-val)/init_val*100),3) if init_val!=0 else "∞"
  return val, delta