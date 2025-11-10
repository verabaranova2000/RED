import numpy as np
import gemmi
import re
import plotly.graph_objects as go
from atoms.scattering_factors.parametric_factors import PARAM


## ===== Кривая рассеяния X-ray =====
def view_X_ray_form_factors(atom_name, curves, check_norm=False, return_fig=False):
  """
  Строит кривую рентгеновского фактора рассеяния для заданного атома, используя
  теоретические данные IT92 и кривые валентных оболочек (Coppens).

  Parameters
  ----------
  atom_name : str                      ← Название атома (например, 'Ca', 'F'), используется для извлечения параметров.
  curves : dict                        ← Словарь с кривыми рассеяния по оболочкам атома; ключ 'neutral atom' обязателен.
                                       ← Каждый элемент должен содержать готовый trace для добавления в Plotly.
  check_norm : bool, default=False     ← Если True, проверяет нормировку коэффициентов IT92 по атомному номеру Z.
  return_fig : bool, default=False     ← Если True, возвращает объект plotly Figure; иначе интерактивно отображает график.

  Returns
  -------
  plotly.graph_objects.Figure or None  ← Объект Figure, если return_fig=True; иначе None.

  Notes
  -----
  - Используются релятивистские рентгеновские факторы рассеяния IT92 для нейтрального атома.
  - Для валентных оболочек используются кривые Coppens (интерполированные и готовые к визуализации).
  - Интерактивная визуализация Plotly с возможностью масштабирования и просмотра легенды.
  - Проверка нормировки (check_norm=True) позволяет убедиться, что сумма коэффициентов равна Z.
  """
  atom  = re.sub("[^A-Za-z]", "", atom_name)
  Z     = PARAM["elements"][atom]['Z']  # атомный номер
  title_of_fig = f'Atom "{atom}" (Z={Z}): Relativistic X-ray Scattering Factors for Atom and Atomic Valence Shells'
  config = {'scrollZoom': True}
  fig=go.Figure()
  ## --- Кривая рентгеновского рассеяния из IT 92 ---
  coef_x=gemmi.Element(atom).it92
  curve_gemmi=np.array([coef_x.calculate_sf(stol2=stl0*stl0) for stl0 in curves['neutral atom']['x']])
  tr_gemmi=go.Scatter(x=curves['neutral atom']['x'], y=curve_gemmi, name='Neutral atom (IT 92)', line=dict(color='yellow'),xaxis="x", yaxis="y")
  fig.add_trace(tr_gemmi)
  ## --- Проверка нормировки ---
  if check_norm==True:
    sum_coeff=sum(coef_x.a)+coef_x.c
    if float(sum_coeff)==float(Z): print('The coefficients is normalized: %s'%Z+' = %s'%sum_coeff)
    else: print('The coefficients is not normalized: %s'%Z+' ≠ %s'%sum_coeff)
  ## --- Кривые, вычисленные в группе Coppens ---
  for k,v in curves.items():
    fig.add_trace(v['trace'])
  fig.update_layout(
    title=dict(text=title_of_fig,
               font=dict(family='Times New Roman'),
                         subtitle=dict(text="Reference: https://harker.chem.buffalo.edu/group/ptable.html",
                         font=dict(color="gray", size=13, family='Times New Roman'))),

    xaxis_title="sin(θ)/λ,  Å<sup>-1</sup>",        xaxis_title_font_family='Times New Roman',
    yaxis_title="Form-factor",              yaxis_title_font_family='Times New Roman',
    dragmode="zoom",
    hovermode="x",
    legend=dict(traceorder="reversed"),
    height=800,
    template="plotly_white",
    margin=dict(t=100, b=100))
  ## --- Результат ---
  if return_fig==False:
    fig.show(config=config)
    res=None
  elif return_fig==True: res=fig
  return res




## ===== Кривая рассеяния electron =====
def view_electron_form_factors(atom_name, prefix_KPhase, curves, experimental_stl=None, return_fig=False, aspherical=False, parametrisations=None, **pars):
  """
  Строит кривую электронного фактора рассеяния для заданного атома с использованием
  теоретических данных IT, каппа-модели и опционально асферической модели.

  Parameters
  ----------
  atom_name : str                          ← Название атома (например, 'Ca', 'F'), используется для извлечения параметров.
  prefix_KPhase : str                      ← Префикс фазы для доступа к параметрам атома в словаре pars.
  curves : dict                            ← Словарь с кривыми рассеяния по оболочкам атома; ключ 'neutral atom' обязателен.
  experimental_stl : array-like, optional  ← Значения sin(θ)/λ для экспериментальных точек, которые будут добавлены на график.
  return_fig : bool, default=False         ← Если True, возвращает объект plotly Figure, иначе выводит интерактивный график.
  aspherical : bool, default=False         ← Применять ли асферическую модель электронных оболочек.
  parametrisations : dict, optional        ← Параметры для асферической модели (коэффициенты a, b и углы).
  **pars : dict                            ← Параметры каппа-модели для расчёта факторов рассеяния (P, kappa и пр.).

  Returns
  -------
  plotly.graph_objects.Figure or None      ← Объект Figure, если return_fig=True; 
                                             иначе None, строит график.
    
  Notes
  -----
  - Кривые рассеяния рассчитываются для нейтрального атома из базы IT, с использованием
    каппа-модели и, при необходимости, асферической поправки.
  - Асферическая модель учитывает валентные оболочки и их угловую зависимость.
  - Используется интерактивная визуализация Plotly с возможностью масштабирования и
    отображения экспериментальных точек.
  """
  from atoms.models import fe, fe_el_kmodel
  atom=re.sub("[^A-Za-z]", "", atom_name)
  title_of_fig='Atom "'+atom+'":  Relativistic electron scattering factors for neutral atom from IT and kappa-model'
  config = {'scrollZoom': True}
  fig=go.Figure()
  ## --- Кривая рассеяния электронов из IT 92 ---
  curve_Gauss = np.array([fe(stl0, element_ID=atom) for stl0 in curves['neutral atom']['x']])
  tr_c4322=go.Scatter(x=curves['neutral atom']['x'], y=curve_Gauss, name='Neutral atom (IT)', line=dict(color='yellow'),xaxis="x", yaxis="y")
  fig.add_trace(tr_c4322)
  ## --- Кривые, вычисленные по каппа-модели ---
  curve_kappa = np.array([fe_el_kmodel(stl0, prefix_KPhase, atom_name, curves, **pars) for stl0 in curves['neutral atom']['x'][1:]])
  tr_kappa = go.Scatter(x=curves['neutral atom']['x'][1:], y=curve_kappa, name='kappa-model',  line=dict(color='#D626FF'),xaxis="x", yaxis="y")
  fig.add_trace(tr_kappa)

  ## --- Добавляем экспериментальные точки stl ---
  if experimental_stl!=None:
    stl_y=[fe_el_kmodel(stl0, prefix_KPhase, atom_name, curves, **pars) for stl0 in experimental_stl]
    tr_hkl_points=go.Scatter(x=experimental_stl, y=stl_y,  name='observed reflections (h,k,l)', mode='markers',
                             line=dict(color='#D626FF'), marker=dict(size=4, symbol="diamond", line=dict(width=1, color="DarkSlateGrey")), xaxis="x", yaxis="y")
    fig.add_trace(tr_hkl_points)

  fig.update_layout(
    title=dict(text=title_of_fig,
               font=dict(family='Times New Roman'),
                         subtitle=dict(text="Valence shells: "+', '.join([k for k,v in curves.items() if k not in ['neutral atom', 'core']]),
                         font=dict(color="gray", size=13, family='Times New Roman'))
                         ),

    xaxis_title="sin(θ)/λ,  Å<sup>-1</sup>",        xaxis_title_font_family='Times New Roman',
    yaxis_title="Form-factor",              yaxis_title_font_family='Times New Roman',
    dragmode="zoom",
    hovermode="x",
    legend=dict(traceorder="reversed"),
    height=800,
    template="plotly_white",
    margin=dict(t=100, b=100))
  
  ## --- Если применяем асферическую добавку: ---
  if aspherical==True and parametrisations!=None:
    from atoms.models import f_ab8, fe_el_asherical
    x=curves['neutral atom']['x']
    for k,v in parametrisations.items():
      name_of_trace='aspherical model: '+k+ ' (normalized)' if k not in ['neutral atom', 'core'] else 'aspherical model: '+k
      trace_k=go.Scatter(x=x, y=f_ab8(stl=x,a=v['a'],b=v['b']), name=name_of_trace)
      fig.add_trace(trace_k)
    ## --- Фактор рассеяния (сумма остова и всех валентных оболочек) ---
    y_sum=fe_el_asherical(x, prefix_KPhase, atom_name, parametrisations, **pars)
    tr_asph_sum=go.Scatter(x=x, y=y_sum, name='aspherical source: core + spherical and aspherical shells')
    fig.add_trace(tr_asph_sum)

    fig.update_layout(
      title=dict(text=title_of_fig,
                 font=dict(family='Times New Roman'),
                           subtitle=dict(text="Valence shells: "+', '.join([k for k,v in curves.items() if k not in ['neutral atom', 'core']])+
                                         '<br>Valence shells (aspherical model): '+', '.join([k for k,v in parametrisations.items() if k not in ['neutral atom', 'core']])+
                                         '<br>Aspherical model and parametrisations from: doi:10.1107/S0021889809033147 ',
                           font=dict(color="gray", size=13, family='Times New Roman'))
                           ))
  if return_fig==False:
    fig.show(config=config)
    res=None
  elif return_fig==True: res=fig
  return res



__all__ = ["view_X_ray_form_factors", "view_electron_form_factors"]