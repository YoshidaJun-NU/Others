import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.signal import savgol_filter, find_peaks

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro (Ultimate)", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer Pro")
st.markdown("標準物質校正、サイクル分割、複数ピーク解析、**詳細な原理解説**を搭載。")

# --- セッション状態の初期化 ---
if 'calibration_shift' not in st.session_state:
    st.session_state['calibration_shift'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False
if 'single_peak_results' not in st.session_state:
    st.session_state['single_peak_results'] = [] # 単独ピーク保存用（必要に応じて使用）
if 'pair_results' not in st.session_state:
    st.session_state['pair_results'] = []

# --- 解説テキスト (更新版) ---
EXPLANATION_TEXT = """
CV測定値からエネルギー準位（HOMO/LUMO）を算出する際の理論的背景と計算式について解説します。

#### 1. 測定原理と基準物質
サイクリックボルタンメトリー（CV）で得られる電位は、参照電極（Ag/Ag+など）に対する相対値です。
物質固有の絶対エネルギー準位（eV）を知るためには、**真空準位（Vacuum Level）** との対応付けが必要です。
そのための「物差し」として、挙動が安定している**フェロセン（$Fc/Fc^+$）**の酸化還元電位を用います。

#### 2. 計算式と定数 (4.8 eV vs 5.1 eV)
フェロセンのフェルミ準位が、真空準位に対してどの深さにあるかについては、主に2つの解釈があります。
研究分野や投稿先の慣習に合わせて使い分けてください。

**(A) 有機エレクトロニクス分野（OLED, OPVなど）**
一般的に **4.8 eV** が採用されます。
$$ E_{HOMO} = -e (E_{ox}^{onset} + 4.8) \\quad [eV] $$
$$ E_{LUMO} = -e (E_{red}^{onset} + 4.8) \\quad [eV] $$
* ここで $E^{onset}$ は、$Fc/Fc^+$ を 0 V とした時の立ち上がり電位です。
* 出典: Pommerehne et al., *Adv. Mater.* **1995**, *7*, 551. など

**(B) 電気化学・物理化学分野**
標準水素電極（SHE）の絶対電極電位（約 -4.44 eV）に基づく厳密な換算として、**5.1 eV** を用いる場合があります。
$$ E_{HOMO} = -e (E_{ox}^{onset} + 5.1) \\quad [eV] $$

#### 3. Onset（立ち上がり）か Peak（ピーク）か？
* **$E_{onset}$ (立ち上がり):** HOMO/LUMOレベルの算出には、一般的にこちらを使います。バンドギャップの端（Band Edge）に対応するためです。
* **$E_{1/2}$ (式量電位):** 酸化還元反応の熱力学的な中心を知りたい場合（標準電極電位の特定など）に使います。

#### 4. バンドギャップ ($E_g$)
$$ E_g = E_{LUMO} - E_{HOMO} \\approx e (E_{ox}^{onset} - E_{red}^{onset}) $$
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
# Tab 1: 校正
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")
    fc_file = st.file_uploader("標準物質 (例: Ferrocene)", type=['csv', 'txt', 'dat'], key="fc_u")
    if fc_file:
        df_fc = load_data(fc_file, skip_rows, sep=data_sep)
        if df_fc is not None and df_fc.shape[1] >= max(x_col_idx, y_col_idx):
            v_fc = df_fc.iloc[:, x_col_idx-1].values
            i_fc = df_fc.iloc[:, y_col_idx-1].values
            if smoothing: i_fc = smooth_data(i_fc)
            
            c_fc1, c_fc2 = st.columns(2)
            min_v, max_v = float(np.min(v_fc)), float(np.max(v_fc))
            with c_fc1: s_min = st.number_input("探索 Min (V)", value=min_v, format="%.2f", key="fc_min")
            with c_fc2: s_max = st.number_input("探索 Max (V)", value=max_v, format="%.2f", key="fc_max")
            
            mask = (v_fc >= s_min) & (v_fc <= s_max)
            v_roi, i_roi = v_fc[mask], i_fc[mask]
            
            if len(v_roi) > 0:
                # 簡易検出
                idx_max, idx_min = np.argmax(i_roi), np.argmin(i_roi)
                E_pa, I_pa = v_roi[idx_max], i_roi[idx_max]
                E_pc, I_pc = v_roi[idx_min], i_roi[idx_min]
                E_half = (E_pa + E_pc)/2
                
                res1, res2, res3 = st.columns(3)
                res1.metric("酸化 Epa", f"{E_pa:.3f} V")
                res2.metric("還元 Epc", f"{E_pc:.3f} V")
                res3.metric("式量電位 E1/2", f"{E_half:.3f} V")
                
                if st.button("基準 (0 V) に設定"):
                    st.session_state['calibration_shift'] = E_half
                    st.session_state['is_calibrated'] = True
                    st.success(f"校正完了: Shift = {E_half:.4f} V")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=v_fc, y=i_fc, mode='lines', name='Raw', line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=[E_pa], y=[I_pa], mode='markers', marker=dict(color='red', size=10), name='Anodic'))
                fig.add_trace(go.Scatter(x=[E_pc], y=[I_pc], mode='markers', marker=dict(color='blue', size=10), name='Cathodic'))
                fig.add_vline(x=E_half, line_dash='dash', line_color='green')
                fig = update_fig_layout(fig, f"Standard ({fc_file.