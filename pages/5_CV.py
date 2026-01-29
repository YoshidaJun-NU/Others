import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.signal import savgol_filter, find_peaks

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro (Ultimate)", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer")
st.markdown("標準物質校正、サイクル分割、複数ピーク解析、解説を搭載。")

# --- セッション状態の初期化 ---
if 'calibration_shift' not in st.session_state:
    st.session_state['calibration_shift'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False
if 'single_peak_results' not in st.session_state:
    st.session_state['single_peak_results'] = []
if 'pair_results' not in st.session_state:
    st.session_state['pair_results'] = []
# 標準物質の一時データ保存用
if 'temp_fc_results' not in st.session_state:
    st.session_state['temp_fc_results'] = None

# --- 解説テキスト ---
EXPLANATION_TEXT = """
CV測定値からエネルギー準位（HOMO/LUMO）を算出する際の理論的背景と計算式について解説します。

#### 1. 測定原理と基準物質
サイクリックボルタンメトリー（CV）で得られる電位は、参照電極（Ag/Ag+など）に対する相対値です。
物質固有の絶対エネルギー準位（eV）を知るためには、**真空準位（Vacuum Level）** との対応付けが必要です。
そのための「物差し」として、挙動が安定している**フェロセン（Fc/Fc+）**の酸化還元電位を用います。

#### 2. 計算式と定数 (4.8 eV vs 5.1 eV)
フェロセンのフェルミ準位が、真空準位に対してどの深さにあるかについては、主に2つの解釈があります。
研究分野や投稿先の慣習に合わせて使い分けてください。

**(A) 有機エレクトロニクス分野（OLED, OPVなど）**
一般的に **4.8 eV** が採用されます。
$$E_{HOMO} = -e (E_{ox}^{onset} + 4.8) \\quad [eV]$$
$$E_{LUMO} = -e (E_{red}^{onset} + 4.8) \\quad [eV]$$
* ここで $E^{onset}$ は、$Fc/Fc^+$ を 0 V とした時の立ち上がり電位です。
* 出典: Pommerehne et al., *Adv. Mater.* **1995**, *7*, 551. など

**(B) 電気化学・物理化学分野**
標準水素電極（SHE）の絶対電極電位（約 -4.44 eV）に基づく厳密な換算として、**5.1 eV** を用いる場合があります。
$$E_{HOMO} = -e (E_{ox}^{onset} + 5.1) \\quad [eV]$$

#### 3. Onset（立ち上がり）か Peak（ピーク）か？
* **$E_{onset}$ (立ち上がり):** HOMO/LUMOレベルの算出には、一般的にこちらを使います。バンドギャップの端（Band Edge）に対応するためです。
* **$E_{1/2}$ (式量電位):** 酸化還元反応の熱力学的な中心を知りたい場合（標準電極電位の特定など）に使います。

#### 4. バンドギャップ ($E_g$)
$$E_g = E_{LUMO} - E_{HOMO} \\approx e (E_{ox}^{onset} - E_{red}^{onset})$$
光学測定（UV-Vis吸収端）から求めた $E_g$ と比較することで、計算の妥当性を検証することが推奨されます。
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
    amplitude = np.max(i) - np.min(i)
    prom = amplitude * prominence_val if amplitude > 0 else None
    peaks_top_idx, _ = find_peaks(i, prominence=prom)
    peaks_top = [{"E": v[idx], "I": i[idx], "Type": "Anodic"} for idx in peaks_top_idx]
    peaks_btm_idx, _ = find_peaks(-i, prominence=prom)
    peaks_btm = [{"E": v[idx], "I": i[idx], "Type": "Cathodic"} for idx in peaks_btm_idx]
    peaks_top.sort(key=lambda x: x["E"])
    peaks_btm.sort(key=lambda x: x["E"])
    return peaks_top, peaks_btm

def split_cycles_by_voltage(v, i, v_init, v_max, v_min):
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
        if search_start >= len(v): cycle_end_idx = len(v)
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
        xaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, linewidth=axis_width, linecolor='black', mirror=show_mirror, ticks="outside" if show_ticks else "", tickwidth=axis_width, tickcolor='black', showticklabels=show_ticks),
        yaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, linewidth=axis_width, linecolor='black', mirror=show_mirror, ticks="outside" if show_ticks else "", tickwidth=axis_width, tickcolor='black', showticklabels=show_ticks)
    )
    return fig

# ==========================================
# サイドバー設定
# ==========================================
st.sidebar.header("📂 データ設定")
with st.sidebar.expander("列番号・フォーマット", expanded=False):
    st.markdown("**デフォルト設定（サンプル用）**")
    c1, c2 = st.columns(2)
    with c1: x_col_idx = st.number_input("横軸 (E) 列", value=2, min_value=1)
    with c2: y_col_idx = st.number_input("縦軸 (I) 列", value=3, min_value=1)
    skip_rows = st.number_input("ヘッダー行数", value=1, min_value=0)
    data_sep = st.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)
    smoothing = st.checkbox("スムージング", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📥 ファイルアップロード")
sample_files = st.sidebar.file_uploader("サンプルデータ選択", type=['csv', 'txt', 'dat'], accept_multiple_files=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📊 グラフ設定", expanded=False):
    line_width = st.slider("線太さ", 0.5, 5.0, 2.0, 0.1)
    color_mode = st.radio("配色", ["自動", "単色"], horizontal=True)
    custom_color = st.color_picker("色指定", "#000000")
    st.markdown("**軸設定**")
    show_grid = st.checkbox("グリッド", value=True)
    show_ticks = st.checkbox("目盛ラベル", value=True)
    show_mirror = st.checkbox("枠線 (Mirror)", value=True)
    axis_width = st.slider("軸太さ", 1.0, 5.0, 2.0, 0.5)
    font_size = st.number_input("フォントサイズ", value=14, min_value=8)

# ==========================================
# タブ構成
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 校正", "2️⃣ 個別解析", "3️⃣ 比較", "📝 HOMO/LUMO", "ℹ️ メモ・原理"
])

# ==========================================
# Tab 1: 校正 (Modified: Collapsible Sections)
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")
    fc_file = st.file_uploader("標準物質 (例: Ferrocene)", type=['csv', 'txt', 'dat'], key="fc_u")
    
    if fc_file:
        # --- 1. ファイルプレビュー機能 (折りたたみ) ---
        with st.expander("📄 ファイルプレビュー (先頭5行)", expanded=True):
            try:
                fc_file.seek(0)
                # 生データを少しだけ読んで表示 (列構造の確認用)
                df_preview = pd.read_csv(fc_file, header=None, nrows=5, sep=data_sep if data_sep != 'auto' else None, engine='python')
                st.dataframe(df_preview, use_container_width=True)
            except Exception as e:
                st.error(f"プレビューを表示できませんでした: {e}")
            fc_file.seek(0) # ポインタを戻す

        # --- 2. 設定フォーム (折りたたみ) ---
        with st.expander("⚙️ 読み込み・解析設定", expanded=True):
            with st.form(key='fc_settings_form'):
                st.markdown("**列・ヘッダー指定**")
                c_set1, c_set2, c_set3 = st.columns(3)
                fc_x_col = c_set1.number_input("横軸 (E) 列", value=x_col_idx, min_value=1, key="fc_x")
                fc_y_col = c_set2.number_input("縦軸 (I) 列", value=y_col_idx, min_value=1, key="fc_y")
                fc_skip = c_set3.number_input("ヘッダー行数", value=skip_rows, min_value=0, key="fc_skip")
                
                st.markdown("**ピーク探索範囲 (V)**")
                c_fc1, c_fc2 = st.columns(2)
                s_min = c_fc1.number_input("探索 Min", value=-1.0, step=0.1)
                s_max = c_fc2.number_input("探索 Max", value=1.0, step=0.1)
                
                st.markdown("---")
                submit_btn = st.form_submit_button("解析実行 / 再プロット")

        # --- 3. 解析ロジック ---
        if submit_btn:
            # データのロード
            df_fc = load_data(fc_file, fc_skip, sep=data_sep)
            
            if df_fc is not None and df_fc.shape[1] >= max(fc_x_col, fc_y_col):
                v_fc = df_fc.iloc[:, fc_x_col-1].values
                i_fc = df_fc.iloc[:, fc_y_col-1].values
                if smoothing: i_fc = smooth_data(i_fc)
                
                # ピーク検出
                mask = (v_fc >= s_min) & (v_fc <= s_max)
                v_roi, i_roi = v_fc[mask], i_fc[mask]
                
                if len(v_roi) > 0:
                    idx_max, idx_min = np.argmax(i_roi), np.argmin(i_roi)
                    E_pa, I_pa = v_roi[idx_max], i_roi[idx_max]
                    E_pc, I_pc = v_roi[idx_min], i_roi[idx_min]
                    E_half = (E_pa + E_pc)/2
                    
                    # 結果をsession_stateに保存
                    st.session_state['temp_fc_results'] = {
                        "v_fc": v_fc, "i_fc": i_fc,
                        "E_pa": E_pa, "I_pa": I_pa,
                        "E_pc": E_pc, "I_pc": I_pc,
                        "E_half": E_half,
                        "filename": fc_file.name
                    }
                else:
                    st.warning("指定範囲内にデータが見つかりません。探索範囲を広げてください。")
                    st.session_state['temp_fc_results'] = None
            else:
                st.error("指定された列番号がデータ範囲外です。プレビューを確認して修正してください。")
                st.session_state['temp_fc_results'] = None

        # --- 4. 結果表示と校正ボタン ---
        if st.session_state['temp_fc_results'] is not None:
            res = st.session_state['temp_fc_results']
            
            st.divider()
            st.markdown("### 📊 解析結果")
            
            # 結果数値
            res1, res2, res3 = st.columns(3)
            res1.metric("酸化 Epa", f"{res['E_pa']:.3f} V")
            res2.metric("還元 Epc", f"{res['E_pc']:.3f} V")
            res3.metric("式量電位 E1/2", f"{res['E_half']:.3f} V")
            
            # 校正ボタン
            if st.button("👉 この値を基準 (0 V) に設定する"):
                st.session_state['calibration_shift'] = res['E_half']
                st.session_state['is_calibrated'] = True
                st.success(f"校正完了: Shift = {res['E_half']:.4f} V")
            
            # グラフ描画
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['v_fc'], y=res['i_fc'], mode='lines', name='Raw', line=dict(color='gray')))
            fig.add_trace(go.Scatter(x=[res['E_pa']], y=[res['I_pa']], mode='markers', marker=dict(color='red', size=10), name='Anodic'))
            fig.add_trace(go.Scatter(x=[res['E_pc']], y=[res['I_pc']], mode='markers', marker=dict(color='blue', size=10), name='Cathodic'))
            fig.add_vline(x=res['E_half'], line_dash='dash', line_color='green')
            fig = update_fig_layout(fig, f"Standard ({res['filename']})", "V", "A", show_grid, show_mirror, show_ticks, axis_width, font_size)
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Tab 2: 個別解析 (ピーク検出 & ペア登録)
# ==========================================
with tab2:
    st.header("サンプル解析と $E_{1/2}$ ペア算出")
    shift = st.session_state['calibration_shift']
    if st.session_state['is_calibrated']: st.success(f"補正値: {shift:.4f} V")
    else: st.warning("未校正")

    if sample_files:
        sel_file = st.selectbox("ファイル選択", sample_files, format_func=lambda x: x.name)
        if sel_file:
            df_s = load_data(sel_file, skip_rows, sep=data_sep)
            if df_s is not None and df_s.shape[1] >= max(x_col_idx, y_col_idx):
                v_full = df_s.iloc[:, x_col_idx-1].values
                i_full = df_s.iloc[:, y_col_idx-1].values
                if smoothing: i_full = smooth_data(i_full)
                v_calib = v_full - shift

                # サイクル分割
                with st.expander("🔄 サイクル分割設定", expanded=False):
                    use_cy = st.checkbox("有効にする")
                    def_init, def_max, def_min = float(v_full[0]), float(np.max(v_full)), float(np.min(v_full))
                    cy1, cy2, cy3 = st.columns(3)
                    c_init = cy1.number_input("初期電圧", value=def_init, format="%.2f")
                    c_max = cy2.number_input("最大電圧", value=def_max, format="%.2f")
                    c_min = cy3.number_input("最小電圧", value=def_min, format="%.2f")
                
                active_v, active_i = v_calib, i_full
                cy_info = "All Data"
                if use_cy:
                    cycles = split_cycles_by_voltage(v_full, i_full, c_init, c_max, c_min)
                    if cycles:
                        opts = ["All"] + [f"Cycle {k+1}" for k in range(len(cycles))]
                        sel_cy = st.selectbox("サイクル", opts)
                        if sel_cy != "All":
                            idx = int(sel_cy.split(" ")[1]) - 1
                            active_v = cycles[idx]["v"] - shift
                            active_i = cycles[idx]["i"]
                            cy_info = sel_cy

                st.divider()
                col_L, col_R = st.columns([1, 1.3])
                
                with col_L:
                    st.subheader("1. ピーク検出")
                    pm, pM = float(np.min(active_v)), float(np.max(active_v))
                    c_p1, c_p2 = st.columns(2)
                    p_min = c_p1.number_input("Min (V)", value=pm, step=0.1, format="%.2f")
                    p_max = c_p2.number_input("Max (V)", value=pM, step=0.1, format="%.2f")
                    prom = st.slider("感度 (Prominence)", 0.0, 0.5, 0.01, 0.005)

                    mask = (active_v >= p_min) & (active_v <= p_max)
                    v_r, i_r = active_v[mask], active_i[mask]
                    d_top, d_btm = [], []
                    if len(v_r) > 0:
                        d_top, d_btm = detect_multiple_peaks(v_r, i_r, prom)
                    
                    st.caption(f"検出: 酸化{len(d_top)} / 還元{len(d_btm)}")

                    st.subheader("2. ペア作成・登録")
                    if not d_top and not d_btm:
                        st.warning("ピークなし")
                    else:
                        c_s1, c_s2 = st.columns(2)
                        ox_map = {f"{p['E']:.3f} V": p for p in d_top}
                        red_map = {f"{p['E']:.3f} V": p for p in d_btm}
                        k_ox = c_s1.selectbox("酸化ピーク", list(ox_map.keys())) if ox_map else None
                        k_red = c_s2.selectbox("還元ピーク", list(red_map.keys())) if red_map else None

                        if k_ox and k_red:
                            s_ox, s_red = ox_map[k_ox], red_map[k_red]
                            val_half = (s_ox['E'] + s_red['E']) / 2
                            st.success(f"**$E_{{1/2}}$ = {val_half:.4f} V**")
                            if st.button("このペアを登録 💾"):
                                st.session_state['pair_results'].append({
                                    "File": sel_file.name, "Cycle": cy_info,
                                    "E_1/2": val_half, "E_pa": s_ox['E'], "E_pc": s_red['E'],
                                    "I_pa": s_ox['I'], "I_pc": s_red['I']
                                })

                with col_R:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=active_v, y=active_i, mode='lines', line=dict(color='black', width=2), name="Current"))
                    fig.add_trace(go.Scatter(x=v_r, y=i_r, mode='lines', line=dict(color='orange', width=4), opacity=0.3, showlegend=False))
                    
                    if d_top:
                        fig.add_trace(go.Scatter(x=[p['E'] for p in d_top], y=[p['I'] for p in d_top], mode='markers', marker=dict(color='red', size=7, symbol='circle-open'), name="Ox Cand."))
                    if d_btm:
                        fig.add_trace(go.Scatter(x=[p['E'] for p in d_btm], y=[p['I'] for p in d_btm], mode='markers', marker=dict(color='blue', size=7, symbol='circle-open'), name="Red Cand."))
                    
                    saved = [p for p in st.session_state['pair_results'] if p['File'] == sel_file.name]
                    for sp in saved:
                        fig.add_vline(x=sp["E_1/2"], line_dash="dot", line_color="green", opacity=0.6)
                        fig.add_trace(go.Scatter(x=[sp["E_pa"], sp["E_pc"]], y=[sp["I_pa"], sp["I_pc"]], mode='markers+lines', marker=dict(color='green', size=10, symbol='star'), line=dict(dash='dot', width=1), name=f"E1/2={sp['E_1/2']:.2f}"))

                    fig = update_fig_layout(fig, f"{sel_file.name} ({cy_info})", "V vs Fc/Fc+", "A", show_grid, show_mirror, show_ticks, axis_width, font_size)
                    st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("📋 登録リスト")
                if st.session_state['pair_results']:
                    st.dataframe(pd.DataFrame(st.session_state['pair_results']), use_container_width=True)
                    if st.button("全削除 🗑️"):
                        st.session_state['pair_results'] = []
                        st.rerun()

# ==========================================
# Tab 3: 比較 (簡易表示)
# ==========================================
with tab3:
    st.header("比較・重ね書き")
    if sample_files:
        comp_data = {}
        for f in sample_files:
            d = load_data(f, skip_rows, sep=data_sep)
            if d is not None and d.shape[1] >= max(x_col_idx, y_col_idx):
                v_r = d.iloc[:, x_col_idx-1].values - shift
                i_r = d.iloc[:, y_col_idx-1].values
                if smoothing: i_r = smooth_data(i_r)
                comp_data[f.name] = {"v": v_r, "i": i_r}
        
        c_o1, c_o2 = st.columns([1, 2])
        sel_fs = c_o1.multiselect("表示ファイル", list(comp_data.keys()), default=list(comp_data.keys()))
        norm = c_o1.checkbox("正規化")
        offset = c_o1.number_input("Yオフセット", value=0.0, format="%.2e")

        if sel_fs:
            fig_c = go.Figure()
            colors = pc.qualitative.Plotly
            for idx, fn in enumerate(sel_fs):
                v_d, i_d = comp_data[fn]["v"], comp_data[fn]["i"]
                if norm: i_d /= np.max(np.abs(i_d)) if np.max(np.abs(i_d)) > 0 else 1
                i_d += offset * idx
                lc = colors[idx % len(colors)] if color_mode == "自動" else custom_color
                fig_c.add_trace(go.Scatter(x=v_d, y=i_d, mode='lines', name=fn, line=dict(color=lc, width=line_width)))
            
            yl = "Normalized I" if norm else "Current A"
            fig_c = update_fig_layout(fig_c, "Comparison", "V vs Fc/Fc+", yl, show_grid, show_mirror, show_ticks, axis_width, font_size)
            st.plotly_chart(fig_c, use_container_width=True)

# ==========================================
# Tab 4: HOMO/LUMO
# ==========================================
with tab4:
    st.header("🧪 HOMO / LUMO")
    c1, c2 = st.columns(2)
    e_ox = c1.number_input("Ox Onset (V)", 0.5)
    ref_lv = c1.number_input("Fc Level (eV)", 4.8)
    c1.metric("HOMO", f"{-(e_ox + ref_lv):.2f} eV")
    e_red = c2.number_input("Red Onset (V)", -1.5)
    c2.metric("LUMO", f"{-(e_red + ref_lv):.2f} eV")

# ==========================================
# Tab 5: メモ
# ==========================================
with tab5:
    st.header("📝 メモ・原理")
    with st.expander("📚 フェロセンの基準エネルギーとHOMO/LUMO計算の詳細", expanded=True):
        st.markdown(EXPLANATION_TEXT)