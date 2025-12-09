# ---- Генерация симметричных атомов и таблица ----

import numpy as np
import pandas as pd
from fractions import Fraction
from IPython.display import HTML, display

def _pbc_dist(a, b):
    """
    Минимальное образное расстояние между fractional coords a и b.
    Расстояние — евклидово ∥d∥
    Это стандартный и канонический способ вычислять расстояния с PBC (периодические граничные условия).
    """
    d = np.array(a) - np.array(b)
    d = d - np.round(d)   # приводим к интервалу [-0.5, 0.5)
    return np.linalg.norm(d)


def get_all_positions_in_cell_for_atom(x,y,z, Operations_symmetry, tolerance=0.005):         # На вход: координаты атома, матрицы операций симметрии
    """
    Генерация всех симметричных положений одного атома в элементарной ячейке.

    Позиции атома размножаются всеми операциями симметрии кристаллической решётки,
    после чего приводятся к диапазону [0, 1) в дробных координатах и исключаются дубликаты,
    считая позиции совпадающими, если расстояние между ними меньше заданной точности.

    Параметры
    ----------
    x, y, z : float
        Дробные координаты атома в исходной ячейке.
    Operations_symmetry : list
        Список операций симметрии кристалла. Каждая операция задаётся как кортеж 
        (трансляция, матрица поворота).
    tolerance : float, optional
        Максимальное расстояние между позициями, при котором они считаются эквивалентными 
        (по умолчанию 0.005).

    Возвращает
    -------
    result : list of np.ndarray
        Список уникальных координат атома, учтённых с периодическими граничными условиями.
    
    Example:
    -------
    ops_sym = get_symmetry_matrix_of_crystal_lattice(cif_data)
    xyz_for_atom = get_all_positions_in_cell_for_atom(0.5, 0.5, 0.5, ops_sym)
    """
    x,y,z,Operations_symmetry=x,y,z,Operations_symmetry
    position      = np.array([x,y,z])                                                # Позиция атома, которую будем размножать операциями симметрии
    all_positions = [position]
    # --- 1. Размножение атомной позиции операциями симметрии ---
    for i in range(len(Operations_symmetry)):                                   # Размножаем атомы всеми операциями симметрии
      new_xyz = Operations_symmetry[i][1].dot(position[..., None])              # Умножение столбца (атомную позицию) на оператор поворота G
      new_xyz = new_xyz.transpose()[0]                                          # Превращаем столбец в строку
      new_xyz = new_xyz+Operations_symmetry[i][0]                               # Прибавляем вектор трансляции
      all_positions.append(new_xyz)
    # --- 2. Приведение всех координат в диапазон [0, 1) ---
    all_positions = [pos % 1.0 for pos in all_positions]
    # --- 3. Исключение дубликатов ---
    result = []
    for pos in all_positions:
        if not any(_pbc_dist(pos, r) < tolerance for r in result):
            result.append(pos)
    return result






# ========= Таблица ===================

def _format_coord(x, max_den=24, tol=1e-5):
    """Пытаемся представить координату как простую дробь (1/3, 3/4 и т.п.), иначе 6 знаков."""
    xf = float(x)
    frac = Fraction(xf).limit_denominator(max_den)
    if abs(float(frac) - xf) < tol:
        # показать как '0' или '3/4' или '1/3'
        if frac.numerator == 0:
            return "0"
        if abs(frac.numerator) == abs(frac.denominator):
            return "1"
        return f"{frac.numerator}/{frac.denominator}"
    return f"{xf:.6f}"

def print_equiv_atoms(XYZ, ineq_idxs, title="Atomic positions (wyckoff placeholders)"):
    """
    Формирует и отображает интерактивную HTML-таблицу атомных позиций с группировкой
    по неэквивалентным атомам.

    Параметры
    ----------
    XYZ : iterable
        Последовательность атомов, где каждый элемент имеет формат
        [x, y, z, element], задающий дробные координаты (в пределах [0, 1))
        и химический символ атома.
    ineq_idxs : iterable of int
        Индексы неэквивалентных атомов, выступающие в роли репрезентантов
        соответствующих групп симметрически эквивалентных позиций.
        Предполагается, что список отсортирован по порядку следования атомов
        в `XYZ`.
    title : str, optional
        Заголовок таблицы. По умолчанию — «Atomic positions (wyckoff placeholders)».

    Описание
    --------
    Функция формирует HTML-таблицу, в которой каждый неэквивалентный атом
    (репрезентант) выводится в виде отдельной строки. Для каждого репрезентанта
    создаётся вложенная таблица, содержащая список всех атомов, отнесённых к его
    группе. Вложенные таблицы сворачиваются/разворачиваются по нажатию кнопки.

    Для каждого атома дополнительно вычисляется минимальное периодическое
    расстояние до репрезентанта в условиях периодических граничных условий (PBC).
    Это позволяет визуально контролировать корректность группировки атомов по
    симметрически эквивалентным позициям.

    Вывод
    -----
    Функция не возвращает значение. Результатом работы является интерактивная
    HTML-таблица, выводимая в среду выполнения (например, Jupyter Notebook).
    """
    # DataFrame
    df = pd.DataFrame(XYZ, columns=["x", "y", "z", "element"])
    coords = df[["x","y","z"]].to_numpy(dtype=float)
    ineq_idxs = list(ineq_idxs)

    # Для каждого элемента: получаем репрезентанты (индексы из ineq_idxs)
    reps_by_elem = {}
    for idx in ineq_idxs:
        el = df.at[idx, "element"]
        reps_by_elem.setdefault(el, []).append(idx)

    # XYZ и ineq получены
    # inequivalent_indices = ineq_idxs
    # для каждого атома определяем, к какой реп-группе он относится
    atom_to_ineq = np.zeros(len(XYZ), dtype=int)
    for g, start in enumerate(ineq_idxs):
        end = ineq_idxs[g+1] if g+1 < len(ineq_idxs) else len(XYZ)
        atom_to_ineq[start:end] = g

    # Для каждого атома присвоим репрезентанту ближайшего представителя того же элемента
    owner = {}  # atom_index -> rep_index
    for aidx in range(len(XYZ)):
        group = atom_to_ineq[aidx]   # реп-группа, к которой принадлежит атом
        rep_idx = ineq_idxs[group]        # единственный реп этого блока
        owner[aidx] = rep_idx


    # Если есть элементы без репрезентанта (маловероятно), назначаем сами себе
    for i in df.index:
        if i not in owner:
            owner[i] = i

    # Для каждого реп собираем список принадлежащих индексов (включая сам реп)
    members_by_rep = {}
    for atom_idx, rep_idx in owner.items():
        members_by_rep.setdefault(rep_idx, []).append(atom_idx)

    # Соберём HTML таблицу
    html = f"""
    <style>
      .ep-table {{ border-collapse: collapse; font-family: Arial, sans-serif; font-size:13px; }}
      .ep-table th, .ep-table td {{ border: 1px solid #d0d0d0; padding: 6px 8px; }}
      .ep-table th {{ background:#f7f7f8; text-align:left; font-weight:600; }}
      .toggle-btn {{
        cursor: pointer; font-weight:600; padding: 4px 8px; border-radius:5px;
        border:1px solid #bbb; background:#fff; font-size:12px;
      }}
      .element-cell button {{
        all: unset; cursor: pointer; display:inline-block; padding:4px 8px; border-radius:5px;
        border:1px solid #bbb; background:#f5f8fb; font-weight:600;
      }}
      .child-row td {{ padding:6px 8px; background:#fbfbfd; }}
      .nested-table {{ border-collapse: collapse; width:100%; }}
      .nested-table th, .nested-table td {{ border: 1px solid #e6e6e6; padding:4px 6px; font-size:12px; }}
      /* compact on narrow screens */
      @media(max-width:700px){{
        .ep-table th, .ep-table td {{ padding:4px 6px; font-size:12px; }}
      }}
    </style>

    <div style="margin-bottom:8px; font-weight:700;">{title}</div>
    <table class="ep-table">
      <thead>
        <tr><th style="width:60px">Wyckoff</th><th style="width:110px">Element</th><th>x</th><th>y</th><th>z</th><th style="width:60px">#</th></tr>
      </thead>
      <tbody>
    """

    # порядок вывода: по порядку ineq_idxs
    for rep_idx in ineq_idxs:
        if rep_idx not in members_by_rep:
            continue
        rep_row = df.iloc[rep_idx]
        members = sorted(members_by_rep[rep_idx])
        # wyckoff заглушка: оставим '-'
        wyckoff = "—"
        # краткие координаты репа
        x_s = _format_coord(rep_row["x"])
        y_s = _format_coord(rep_row["y"])
        z_s = _format_coord(rep_row["z"])
        # строка видимая
        uid = f"rep_{rep_idx}"
        html += f"""
        <tr>
          <td>{wyckoff}</td>
          <td class="element-cell">
            <button onclick="toggleChild('{uid}')">{rep_row['element']}</button>
          </td>
          <td>{x_s}</td>
          <td>{y_s}</td>
          <td>{z_s}</td>
          <td style="text-align:center">{len(members)}</td>
        </tr>
        """
        # строка с вложенной таблицей (скрытая по умолчанию)
        # вложенная таблица: индекс | x y z | distance-to-rep (periodic)
        child_html = f"""
        <tr id="{uid}" class="child-row" style="display:none">
          <td colspan="6">
            <table class="nested-table">
              <thead><tr><th style="width:40px">#</th><th style="width:45px">idx</th><th>element</th><th>x</th><th>y</th><th>z</th><th style="width:90px">Δ to rep</th></tr></thead>
              <tbody>
        """
        for i, aidx in enumerate(members, start=1):
            r = df.iloc[aidx]
            ds = _pbc_dist(coords[aidx], coords[rep_idx])
            child_html += f"<tr><td style='text-align:center'>{i}</td><td style='text-align:center'>{aidx}</td><td>{r['element']}</td>"
            child_html += f"<td>{_format_coord(r['x'])}</td><td>{_format_coord(r['y'])}</td><td>{_format_coord(r['z'])}</td>"
            child_html += f"<td style='text-align:right'>{ds:.6f}</td></tr>"

        child_html += "</tbody></table></td></tr>"
        html += child_html

    html += "</tbody></table>"

    # JS для переключения
    html += """
    <script>
    function toggleChild(id){
      var el = document.getElementById(id);
      if (!el) return;
      el.style.display = (el.style.display === "none" || el.style.display === "") ? "table-row" : "none";
      // плавная прокрутка к элементу если раскрываем
      if (el.style.display !== "none") {
        setTimeout(()=> el.scrollIntoView({behavior:"smooth", block:"nearest"}), 50);
      }
    }
    </script>
    """

    display(HTML(html))
