import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.signal import savgol_filter, find_peaks

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro (Multi-Pair)", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer")
st.markdown("複数ピーク検出、**任意のピークペアによる $E_{1/2}$ 算出**、サイクル分割に対応。")

# --- セッション状態の初期化 ---
if 'calibration_shift' not in st.session_state:
    st.session_state['calibration_shift'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False
# 保存用リストを2つに分離（単独ピーク用 / 計算ペア用）
if 'single_peak_results' not in st.session_state:
    st.session_state['single_peak_results'] = []
if 'pair_results' not in st.session_state:
    st.session_state['pair_results'] = []

# --- 解説テキスト ---
EXPLANATION_TEXT = """
### 📚 複数ペアの $E_{1/2}$ 算出について

1つのCV曲線に複数の酸化還元反応が含まれる場合（例：第1酸化、第2酸化...）、それぞれの反応に対応する $E_{pa}$（酸化ピーク）と $E_{pc}$（還元ピーク）を正しく組み合わせる必要があります。

このツールでは以下の手順で複数の $E_{1/2}$ を算出できます：
1. **ピーク検出**: 自動で極大・極小点をすべて拾い上げます。
2. **ペアリング**: 検出された候補の中から、対応する酸化・還元ピークをドロップダウンで選択します。
3. **登録**: 「ペア登録」ボタンを押すと、その組み合わせの $E_{1/2}$ が計算され、リストに追加されます。これを必要な回数繰り返してください。
"""

# --- 関数定義 ---
def load_data(uploaded_file, skip_rows, encoding='utf-8', sep='auto'):
    try:
        uploaded_file.seek(0)
        if sep == 'auto':
            try:
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, engine='python', encoding=encoding)
                if df.shape[1] <= 1:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python', encoding=encoding)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python', encoding=encoding)
        else:
            df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=sep, engine='python', encoding=encoding)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except Exception:
        return None

def smooth_data(y, window_length=11, polyorder=3):
    try:
        if window_length > len(y): window_length = len(y) // 2 * 2 + 1 
        return savgol_filter(y, window_length, polyorder)
    except:
        return y

def detect_multiple_peaks(v, i, prominence_val=0.0):
    """指定範囲内のデータから複数のピークを検索してリストで返す"""
    amplitude = np.max(i) - np.min(i)
    prom = amplitude * prominence_val if amplitude > 0 else None

    # 上に凸 (Maxima)
    peaks_top_idx, _ = find_peaks(i, prominence=prom)
    peaks_top = [{"E": v[idx], "I": i[idx], "Type": "Anodic"} for idx in peaks_top_idx]

    # 下に凸 (Minima) -> -i に対して検索
    peaks_btm_idx, _ = find_peaks(-i, prominence=prom)
    peaks_btm = [{"E": v[idx], "I": i[idx], "Type": "Cathodic"} for idx in peaks_btm_idx]

    # Eの値順にソート
    peaks_top.sort(key=lambda x: x["E"])
    peaks_btm.sort(key=lambda x: x["E"])

    return peaks_top, peaks_btm

def split_cycles_by_voltage(v, i, v_init, v_max, v_min):
    """電圧の折り返し点に基づいてサイクルを分割する簡易ロジック"""
    peaks_high, _ = find_peaks(v, height=v_max - abs(v_max)*0.1)
    peaks_low, _ = find_peaks(-v, height=-(v_min + abs(v_min)*0.1))

    n_cycles = min(len(peaks_high), len(peaks_low))
    if n_cycles == 0: return [{"v": v, "i": i}]

    cycles = []
    cycle_start_idx = 0
    for k in range(n_cycles):
        p_h = peaks_high[k]
        p_l = peaks_low[k]
        last_extremum_idx = max(p_h, p_l)
        search_start = last_extremum_idx + 10
        if search_start >= len(v):
            cycle_end_idx = len(v)
        else:
            diff = np.abs(v[search_start:] - v_init)
            local_min_idx = np.argmin(diff)
            cycle_end_idx = search_start + local_min_idx + 1
        
        if cycle_end_idx > len(v): cycle_end_idx = len(v)
        cycles.append({"v": v[cycle_start_idx:cycle_end_idx], "i": i[cycle_start_idx:cycle_end_idx]})
        cycle_start_idx = cycle_end_idx
        if cycle_start_idx >= len(v) - 10: break

    if len(cycles) == 0: return [{"v": v, "i": i}]
    return cycles

def update_fig_layout(fig, title, x_title, y_title, show_grid, show_mirror, show_ticks, axis_width, font_size):
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        height=600, font=dict(size=font_size), hovermode="closest", template="simple_white",
        xaxis=dict(
            showgrid=show_grid, gridcolor='lightgray', 
            showline=True, linewidth=axis_width, linecolor='black',
            mirror=show_mirror, 
            ticks="outside" if show_ticks else "", tickwidth=axis_width, tickcolor='black',
            showticklabels=show_ticks
        ),
        yaxis=dict(
            showgrid=show_grid, gridcolor='lightgray', 
            showline=True, linewidth=axis_width, linecolor='black',
            mirror=show_mirror, 
            ticks="outside" if show_ticks else "", tickwidth=axis_width, tickcolor='black',
            showticklabels=show_ticks
        )
    )
    return fig

# ==========================================
# サイドバー設定
# ==========================================
st.sidebar.header("📂 データ設定")

with st.sidebar.expander("列番号・フォーマット設定", expanded=False):
    col1, col2 = st.columns(2)
    with col1: x_col_idx = st.number_input("横軸 (E/V) 列", value=2, min_value=1)
    with col2: y_col_idx = st.number_input("縦軸 (I/A) 列", value=3, min_value=1)
    skip_rows = st.number_input("ヘッダー行数", value=1, min_value=0)
    data_sep = st.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)
    smoothing = st.checkbox("スムージング (ノイズ除去)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 サンプルデータ")
sample_files = st.sidebar.file_uploader(
    "解析・比較したいファイルをアップロード", 
    type=['csv', 'txt', 'dat'], 
    accept_multiple_files=True, 
    key="sample_upload_sidebar"
)

st.sidebar.markdown("---")
with st.sidebar.expander("📊 グラフ表示設定", expanded=False):
    line_width = st.slider("プロット線太さ", 0.5, 5.0, 2.0, 0.1)
    color_mode = st.radio("配色", ["自動", "単色"], horizontal=True)
    custom_color = st.color_picker("単色指定", "#000000")
    
    st.markdown("**軸・グリッド**")
    show_grid = st.checkbox("グリッド線", value=True)
    show_ticks = st.checkbox("目盛ラベル", value=True)
    show_mirror = st.checkbox("枠線 (Mirror)", value=True)
    axis_width = st.slider("軸・目盛線太さ", 1.0, 5.0, 2.0, 0.5)
    font_size = st.number_input("フォントサイズ", value=14, min_value=8)

# ==========================================
# タブ構成
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 校正", 
    "2️⃣ 個別解析 (ペア算出)", 
    "3️⃣ 比較・重ね書き", 
    "📝 HOMO/LUMO", 
    "ℹ️ メモ・原理"
])

# ==========================================
# Tab 1: 校正
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")
    fc_file = st.file_uploader("標準物質データ (例: Ferrocene)", type=['csv', 'txt', 'dat'], key="fc_upload")

    if fc_file:
        df_fc = load_data(fc_file, skip_rows, sep=data_sep)
        max_col = max(x_col_idx, y_col_idx)
        if df_fc is not None and df_fc.shape[1] >= max_col:
            volt = df_fc.iloc[:, x_col_idx - 1].values
            curr = df_fc.iloc[:, y_col_idx - 1].values
            if smoothing: curr = smooth_data(curr)

            col_r1, col_r2 = st.columns(2)
            min_v, max_v = float(np.min(volt)), float(np.max(volt))
            with col_r1: search_min = st.number_input("探索 Min (V)", value=min_v, format="%.2f", key="fc_min")
            with col_r2: search_max = st.number_input("探索 Max (V)", value=max_v, format="%.2f", key="fc_max")

            mask = (volt >= search_min) & (volt <= search_max)
            v_roi, c_roi = volt[mask], curr[mask]

            if len(v_roi) > 0:
                E_pa, I_pa = v_roi[np.argmax(c_roi)], np.max(c_roi)
                E_pc, I_pc = v_roi[np.argmin(c_roi)], np.min(c_roi)
                E_half = (E_pa + E_pc) / 2

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("酸化 $E_{pa}$", f"{E_pa:.3f} V")
                col_res2.metric("還元 $E_{pc}$", f"{E_pc:.3f} V")
                col_res3.metric("式量電位 $E_{1/2}$", f"{E_half:.3f} V")

                if st.button("この値を基準 (0 V) に設定する"):
                    st.session_state['calibration_shift'] = E_half
                    st.session_state['is_calibrated'] = True
                    st.success(f"校正完了！ シフト値: {E_half:.4f} V")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=volt, y=curr, mode='lines', name='Raw', line=dict(color=custom_color, width=line_width)))
                fig.add_trace(go.Scatter(x=[E_pa], y=[I_pa], mode='markers', name='Anodic', marker=dict(color='red', size=10)))
                fig.add_trace(go.Scatter(x=[E_pc], y=[I_pc], mode='markers', name='Cathodic', marker=dict(color='blue', size=10)))
                fig.add_vline(x=E_half, line_dash="dash", line_color="green")
                fig = update_fig_layout(fig, f"Standard ({fc_file.name})", "V", "A", show_grid, show_mirror, show_ticks, axis_width, font_size)
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Tab 2: 個別解析 (ペアリング機能強化)
# ==========================================
with tab2:
    st.header("サンプル解析と $E_{1/2}$ ペア算出")
    
    shift_val = st.session_state['calibration_shift']
    if st.session_state['is_calibrated']:
        st.success(f"✅ 現在の補正値: **{shift_val:.4f} V**")
    else:
        st.warning("⚠️ 未校正 (元の電圧表示)")

    if sample_files:
        selected_file_obj = st.selectbox("解析するファイルを選択", sample_files, format_func=lambda x: x.name)
        
        if selected_file_obj:
            # データ読み込み
            df_s = load_data(selected_file_obj, skip_rows, sep=data_sep)
            if df_s is not None and df_s.shape[1] >= max(x_col_idx, y_col_idx):
                v_full = df_s.iloc[:, x_col_idx - 1].values
                i_full = df_s.iloc[:, y_col_idx - 1].values
                if smoothing: i_full = smooth_data(i_full)
                v_full_calib = v_full - shift_val

                # --- サイクル分割 ---
                with st.expander("🔄 サイクル分割設定 (必要な場合のみ)", expanded=False):
                    use_cycles = st.checkbox("サイクル分割モード", value=False)
                    def_init, def_max, def_min = float(v_full[0]), float(np.max(v_full)), float(np.min(v_full))
                    c1, c2, c3 = st.columns(3)
                    with c1: c_init = st.number_input("初期電圧", value=def_init, format="%.2f")
                    with c2: c_max = st.number_input("最大電圧", value=def_max, format="%.2f")
                    with c3: c_min = st.number_input("最小電圧", value=def_min, format="%.2f")

                active_v_calib = v_full_calib
                active_i = i_full
                cycle_info_str = "All Data"

                if use_cycles:
                    cycles_data = split_cycles_by_voltage(v_full, i_full, c_init, c_max, c_min)
                    if len(cycles_data) > 0:
                        cy_options = [f"Cycle {k+1}" for k in range(len(cycles_data))]
                        cy_options.insert(0, "All Cycles")
                        selected_cy = st.selectbox("表示サイクル", cy_options)
                        if selected_cy != "All Cycles":
                            idx = int(selected_cy.split(" ")[1]) - 1
                            active_v_calib = cycles_data[idx]["v"] - shift_val
                            active_i = cycles_data[idx]["i"]
                            cycle_info_str = selected_cy
                
                st.divider()
                
                # --- レイアウト分割 ---
                col_main_L, col_main_R = st.columns([1, 1.2])

                # --- 左カラム: ピーク検出とペアリング操作 ---
                with col_main_L:
                    st.subheader("1. ピーク検出 & ペア作成")
                    
                    # 探索設定
                    p_min_def, p_max_def = float(np.min(active_v_calib)), float(np.max(active_v_calib))
                    col_p1, col_p2 = st.columns(2)
                    with col_p1: p_min = st.number_input("Min (V)", value=p_min_def, step=0.1, format="%.2f")
                    with col_p2: p_max = st.number_input("Max (V)", value=p_max_def, step=0.1, format="%.2f")
                    prom_val = st.slider("検出感度 (Prominence)", 0.0, 0.5, 0.01, 0.005)

                    # 検出実行
                    mask_range = (active_v_calib >= p_min) & (active_v_calib <= p_max)
                    v_roi = active_v_calib[mask_range]
                    i_roi = active_i[mask_range]
                    
                    detected_top = []
                    detected_btm = []
                    if len(v_roi) > 0:
                        detected_top, detected_btm = detect_multiple_peaks(v_roi, i_roi, prom_val)

                    st.info(f"検出結果: 酸化 {len(detected_top)}個 / 還元 {len(detected_btm)}個")

                    # --- ペアリング計算ツール ---
                    st.markdown("### 🔗 ペアリングと登録")
                    st.caption("検出されたピークから酸化・還元を1つずつ選び、ペアを作成します。")

                    # 選択用辞書の作成 (表示文字列 -> データオブジェクト)
                    # 見つからない場合はNoneを扱う
                    if not detected_top and not detected_btm:
                        st.warning("ピークが見つかりません。感度や範囲を調整してください。")
                    
                    # 選択UI
                    col_sel1, col_sel2 = st.columns(2)
                    
                    # 酸化側プルダウン
                    ox_map = {f"{p['E']:.4f} V": p for p in detected_top}
                    ox_key = col_sel1.selectbox("🔴 酸化ピーク ($E_{pa}$)", options=list(ox_map.keys())) if ox_map else None
                    
                    # 還元側プルダウン
                    red_map = {f"{p['E']:.4f} V": p for p in detected_btm}
                    red_key = col_sel2.selectbox("🔵 還元ピーク ($E_{pc}$)", options=list(red_map.keys())) if red_map else None

                    # 計算と登録ボタン
                    if ox_key and red_key:
                        sel_ox = ox_map[ox_key]
                        sel_red = red_map[red_key]
                        
                        calc_e_half = (sel_ox['E'] + sel_red['E']) / 2
                        st.markdown(f"**算出 $E_{1/2}$ = {calc_e_half:.4f} V**")
                        
                        if st.button("このペアを登録する 💾", type="primary"):
                            st.session_state['pair_results'].append({
                                "File": selected_file_obj.name,
                                "Cycle": cycle_info_str,
                                "E_pa (V)": sel_ox['E'],
                                "E_pc (V)": sel_red['E'],
                                "E_1/2 (V)": calc_e_half,
                                "I_pa (A)": sel_ox['I'],
                                "I_pc (A)": sel_red['I']
                            })
                            st.success("登録しました！別のペアを選択して再度登録できます。")
                    else:
                        st.caption("酸化・還元の両方が選択されると計算ボタンが表示されます。")

                # --- 右カラム: グラフ表示 ---
                with col_main_R:
                    fig_check = go.Figure()
                    
                    # 元データ
                    fig_check.add_trace(go.Scatter(x=active_v_calib, y=active_i, mode='lines', line=dict(color='black', width=2), name="Current Data"))
                    
                    # 探索範囲
                    fig_check.add_trace(go.Scatter(x=v_roi, y=i_roi, mode='lines', line=dict(color='orange', width=4), opacity=0.3, name="Range", showlegend=False))
                    
                    # 検出ピークプロット
                    if detected_top:
                        fig_check.add_trace(go.Scatter(
                            x=[p['E'] for p in detected_top], 
                            y=[p['I'] for p in detected_top], 
                            mode='markers', marker=dict(color='red', size=8, symbol='circle-open'), name="Detected Ox"
                        ))
                    if detected_btm:
                        fig_check.add_trace(go.Scatter(
                            x=[p['E'] for p in detected_btm], 
                            y=[p['I'] for p in detected_btm], 
                            mode='markers', marker=dict(color='blue', size=8, symbol='circle-open'), name="Detected Red"
                        ))
                    
                    # 登録済みペアの可視化 (E1/2ライン)
                    saved_pairs = [p for p in st.session_state['pair_results'] if p['File'] == selected_file_obj.name]
                    for sp in saved_pairs:
                        fig_check.add_vline(x=sp["E_1/2 (V)"], line_dash="dot", line_color="green", opacity=0.7)
                        # ペアを結ぶ線など（オプション）
                        fig_check.add_trace(go.Scatter(
                            x=[sp["E_pa (V)"], sp["E_pc (V)"]],
                            y=[sp["I_pa (A)"], sp["I_pc (A)"]],
                            mode='markers+lines', marker=dict(color='green', size=10, symbol='star'), 
                            line=dict(color='green', width=1, dash='dot'),
                            name=f"Pair ({sp['E_1/2 (V)']:.2f}V)"
                        ))

                    fig_check = update_fig_layout(fig_check, f"Analysis: {selected_file_obj.name}", "V vs Fc/Fc+", "Current / A", show_grid, show_mirror, show_ticks, axis_width, font_size)
                    st.plotly_chart(fig_check, use_container_width=True)

                # --- 保存リスト表示 ---
                st.divider()
                st.subheader("📋 登録された酸化還元ペアリスト ($E_{1/2}$)")
                
                if st.session_state['pair_results']:
                    res_df = pd.DataFrame(st.session_state['pair_results'])
                    # 表示カラム順序の整理
                    cols = ["File", "Cycle", "E_1/2 (V)", "E_pa (V)", "E_pc (V)", "I_pa (A)", "I_pc (A)"]
                    st.dataframe(res_df[cols], use_container_width=True)
                    
                    if st.button("リストをクリア 🗑️"):
                        st.session_state['pair_results'] = []
                        st.rerun()
                else:
                    st.info("まだペアが登録されていません。")

    else:
        st.info("👈 サイドバーからデータをアップロードしてください。")

# ==========================================
# Tab 3: 比較・重ね書き
# ==========================================
with tab3:
    st.header("📊 データの比較・重ね書き")
    if sample_files:
        data_cache_comp = {}
        for s_file in sample_files:
            df_s = load_data(s_file, skip_rows, sep=data_sep)
            if df_s is not None and df_s.shape[1] >= max(x_col_idx, y_col_idx):
                v_raw = df_s.iloc[:, x_col_idx - 1].values
                i_raw = df_s.iloc[:, y_col_idx - 1].values
                if smoothing: i_raw = smooth_data(i_raw)
                data_cache_comp[s_file.name] = {"v": v_raw - st.session_state['calibration_shift'], "i": i_raw}

        col_opt1, col_opt2 = st.columns([1, 2])
        with col_opt1:
            st.subheader("設定")
            selected_files = st.multiselect("表示ファイル", options=list(data_cache_comp.keys()), default=list(data_cache_comp.keys()))
            normalize = st.checkbox("最大値で正規化", value=False)
            y_offset = st.number_input("Y軸オフセット", value=0.0, step=1e-6, format="%.2e")
            
        with col_opt2:
            st.subheader("プレビュー")
            if selected_files:
                fig_comp = go.Figure()
                colors = pc.qualitative.Plotly
                for idx, fname in enumerate(selected_files):
                    v_dat = data_cache_comp[fname]["v"]
                    i_dat = data_cache_comp[fname]["i"]
                    if normalize:
                        max_val = np.max(np.abs(i_dat))
                        if max_val > 0: i_dat = i_dat / max_val
                    i_dat = i_dat + (y_offset * idx)

                    line_c = colors[idx % len(colors)] if color_mode == "自動" else custom_color
                    fig_comp.add_trace(go.Scatter(
                        x=v_dat, y=i_dat, mode='lines', name=fname,
                        line=dict(color=line_c, width=line_width)
                    ))

                y_label = "Normalized Current / a.u." if normalize else "Current / A"
                fig_comp = update_fig_layout(fig_comp, "Comparison Plot", "V vs Fc/Fc+", y_label, show_grid, show_mirror, show_ticks, axis_width, font_size)
                fig_comp.add_vline(x=0, line_color="gray", line_width=1)
                fig_comp.add_hline(y=0, line_color="gray", line_width=1)
                st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("👈 データなし")

# ==========================================
# Tab 4: HOMO/LUMO
# ==========================================
with tab4:
    st.header("🧪 HOMO / LUMO レベルの算出")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("HOMO")
        e_ox = st.number_input("Oxidation Onset (V)", value=0.5, step=0.01)
        fc_lv = st.number_input("Fc Level (eV)", value=4.8, step=0.1)
        st.metric("HOMO", f"{-(e_ox + fc_lv):.2f} eV")
    with col_c2:
        st.subheader("LUMO")
        e_red = st.number_input("Reduction Onset (V)", value=-1.5, step=0.01)
        st.metric("LUMO", f"{-(e_red + fc_lv):.2f} eV")

# ==========================================
# Tab 5: メモ
# ==========================================
with tab5:
    st.header("📝 メモ・原理")
    st.markdown(EXPLANATION_TEXT)