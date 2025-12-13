import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.plotting_axes import align_zero_axes, symmetric_range_from_arrays


#==== Построение радиальных функций P и Q до и после обработки ====
def show_refined_rwfn_data(orbital_names, r_old, P_old_dict, Q_old_dict,
                             r_new, P_new_dict, Q_new_dict,
                             initial_orb=None,
                             width=820, height=420):
    """
    Визуализация радиальных компонент P(r) и Q(r) релятивистских атомных орбиталей
    до и после численной предобработки.

    Исходные радиальные сетки r, набор орбиталей orbital_names и соответствующие
    компоненты P(r), Q(r) получены из релятивистского атомного расчёта
    (файлы GRASP, чтение с помощью функции read_rwfn_plot).

    Улучшенные радиальные сетки r_new и компоненты P_new(r), Q_new(r) являются
    результатом численной предобработки (сгущение сетки и асимптотическое
    продолжение), реализованной в функции refine_rwfn_data.

    Визуализация используется для контроля устойчивости радиальных функций
    и выявления возможных численных артефактов перед вычислением электронной
    плотности и атомных факторов рассеяния.
    
    Пример вызова:
    --------
    fig = show_refined_rwfn_data(orbital_names, r, P_dict, Q_dict, r2, P2, Q2)
    fig.show()
    """
    if initial_orb is None:
        initial_orb = orbital_names[0]
    # Все трассы в одном списке: P_old, P_new, Q_old, Q_new — для КАЖДОЙ орбитали
    traces = []
    keys = list(orbital_names)

    for nm in keys:
        # P refined (левая ось)
        traces.append(go.Scatter(x=r_new, y=P_new_dict[nm], mode='lines',
                      name=f"{nm} P (refined)", visible=False, line=dict(width=2, color='#CC79A7'),
                      hovertemplate="r=%{x:.4f}<br>P=%{y:.5g}"))  
        # P old (левая ось)
        traces.append(go.Scatter(x=r_old, y=P_old_dict[nm], mode='lines+markers',
                      marker=dict(size=4, color='#7A3E63'), name=f"{nm} P (old)", visible=False,
                      line=dict(width=1, dash='dot', color='#7A3E63'), hovertemplate="r=%{x:.4f}<br>P=%{y:.5g}"))
        # Q refined (правая ось)
        traces.append(go.Scatter( x=r_new, y=Q_new_dict[nm], mode='lines',
                      name=f"{nm} Q (refined)", visible=False,
                      line=dict(width=2, color='#0072B2'), yaxis='y2',   # width=2, dash='dash'
                      hovertemplate="r=%{x:.4f}<br>Q=%{y:.5g}"))        
        # Q old (правая ось)
        traces.append(go.Scatter(x=r_old, y=Q_old_dict[nm],  mode='lines+markers',
                      marker=dict(size=4, color='#003B73'), name=f"{nm} Q (old)", visible=False,
                      line=dict(width=1, dash='dot', color='#003B73'), yaxis='y2',   # (width=1, dash='dot')
                      hovertemplate="r=%{x:.4f}<br>Q=%{y:.5g}"))

    # === ФУНКЦИЯ включения только 4 трасс нужной орбитали ===
    def visibility_for(idx):
        n = len(keys)
        vis = [False] * (4 * n)
        base = idx * 4
        for k in range(4):
            vis[base + k] = True
        return vis

    try:
        init_idx = keys.index(initial_orb)
    except ValueError:
        init_idx = 0

    # Dropdown меню
    buttons = []
    for i, nm in enumerate(keys):
        vis = visibility_for(i) + [True, True]   # +2 — это r_grid traces на нижнем subplot'e
        # --- соберём массивы, по которым считаем диапазоны ---
        left_arrays   = [P_old_dict[nm], P_new_dict[nm]]
        right_arrays  = [Q_old_dict[nm], Q_new_dict[nm]]
        y_left_range  = symmetric_range_from_arrays(left_arrays, pad=1.05)
        y_right_range = symmetric_range_from_arrays(right_arrays, pad=1.05)
        # --- Формируем dict для layout — в plotly для update вторым аргументом даём layout-изменения
        layout_update = {"title": f"{nm}: P(r), Q(r) — old vs refined"}
        if y_left_range is not None:
            layout_update["yaxis.range"] = y_left_range
        if y_right_range is not None:
            layout_update["yaxis2.range"] = y_right_range
        buttons.append(dict(label=nm,
                            method="update",
                            args=[{"visible": vis}, layout_update]))

    # === СБОРКА ФИГУРЫ (2 ряда: большой график + узкий "r-grid" внизу) ===
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.82, 0.08], vertical_spacing=0.02,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],)
    # --- добавляем все основные трассы (P/Q) в верхний ряд ---
    for trace in traces:
        if 'yaxis' in trace.to_plotly_json() and trace.to_plotly_json()['yaxis'] == 'y2':
            fig.add_trace(trace, row=1, col=1, secondary_y=True)
        else:
            fig.add_trace(trace, row=1, col=1, secondary_y=False)
    # --- добавляем нижний подграфик — плотность точек r ---
    fig.add_trace(go.Scatter( x=r_old, y=[0.6]*len(r_old), mode='markers',
                  marker=dict(symbol='line-ns-open', size=6, color='#C7C7C7'),   #'#666666'),
                  showlegend=False, name='r_old', hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=r_new, y=[0.4]*len(r_new), mode='markers',
                  marker=dict(symbol='line-ns-open', size=6, color='#5A5A5A'),    #'#1f77b4'),
                  showlegend=False, name='r_new', hoverinfo='skip'), row=2, col=1)
    # --- Аннотации для r-сеток на нижнем подграфе ---
    fig.add_annotation(xref="paper", yref="paper", x=-0.14, y=0.05,
                        text="<b>r-grid (original)</b>",
                        showarrow=False,
                        font=dict(size=11, color = '#C7C7C7'),
                        align="left")
    fig.add_annotation(xref="paper", yref="paper", x=-0.14, y=0.005,
                       text="<b>r-grid  (refined)</b>",
                       showarrow=False,
                       font=dict(size=11, color='#5A5A5A'),
                       align="left")
    # --- Включаем начальную орбиталь (для основных трасс — первые len(traces) элементов)
    init_vis = visibility_for(init_idx)
    for k, v in enumerate(init_vis):
        fig.data[k].visible = v
    # --- ALIGN ZERO LINES: выставляем симметричные диапазоны по модулю для левой и правой осей ---
    align_zero_axes(fig, row=1, col=1, pad=1.05)

    # Layout — стиль
    fig.update_layout(
        template="simple_white", width=width, height=height,
        margin=dict(l=60, r=80, t=50, b=60),
        title=f"{keys[init_idx]}: P(r), Q(r) — old vs refined",
        updatemenus=[dict(type="dropdown",
                          active=init_idx,
                          x=0.4, y=1.15, xanchor="left", yanchor="top",
                          buttons=buttons )],
        hovermode="x unified",
        font=dict(family="DejaVu Sans, Arial", size=12))
    # --- Настройки осей ---
    fig.update_xaxes(title="", tickformat=".3f")  # r (Bohr)
    # --- левая ось ---
    fig.update_yaxes(title="P(r), a.u.",
                     tickformat=".3e", showgrid=True,
                     gridcolor="#ddd", zeroline=False)
    # --- правая ось (вторичная) ---
    fig.update_layout(yaxis2=dict(title="Q(r), a.u.",
                                  tickformat=".3e", overlaying="y",
                                  side="right", showgrid=False))
    # --- Легенда ---
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                  y=-0.25, xanchor="left", x=0))
    fig.update_yaxes(visible=False, row=2, col=1)      # скрываем вертикальную ось у нижнего графика (чтобы не мешала)
    fig.update_xaxes(title="r (Bohr)", row=2, col=1)
    return fig
