import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.signal import savgol_filter

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer Pro")
st.markdown("標準物質による校正、ピーク解析、**複数データの比較作図**が可能です。")

# --- セッション状態の初期化 ---
if 'calibration_shift' not in st.session_state:
    st.session_state['calibration_shift'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False
if 'peak_results' not in st.session_state:
    st.session_state['peak_results'] = []

# --- 解説テキスト ---
EXPLANATION_TEXT = """
### 📚 フェロセンの基準エネルギーとHOMO/LUMO計算

CV測定からHOMO/LUMOレベルを算出する際、基準物質（フェロセン: $Fc/Fc^+$）のエネルギー準位を真空準位に対してどう定義するかで、計算結果（eV）が変わります。

#### 1. よく使われる値：4.8 eV
有機エレクトロニクス分野（OLEDやOPVなど）では、フェロセンの準位を真空準位から **-4.8 eV** とする以下の式が広く用いられます。
$$ E_{HOMO} = -e (E_{ox}^{onset} + 4.8) \ [eV] $$
(Pommerehne et al., *Adv. Mater.* 7, 551 (1995))

#### 2. もう一つの値：5.1 eV
電気化学の標準電極電位（SHE $\\approx$ -4.44 eV）に基づくと、フェロセンは約 5.1 eV と解釈されることもあります。
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

def update_fig_layout(fig, title, x_title, y_title, show_grid, show_mirror, show_ticks, font_size):
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title,
        height=600, font=dict(size=font_size), hovermode="closest", template="simple_white",
        xaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, mirror=show_mirror, ticks="outside" if show_ticks else "", showticklabels=show_ticks),
        yaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, mirror=show_mirror, ticks="outside" if show_ticks else "", showticklabels=show_ticks)
    )
    return fig

# ==========================================
# サイドバー設定 (データ読み込み & アップロード)
# ==========================================
st.sidebar.header("📂 データ設定")

# 1. カラム設定
with st.sidebar.expander("列番号・フォーマット設定", expanded=False):
    col1, col2 = st.columns(2)
    with col1: x_col_idx = st.number_input("横軸 (E/V) 列", value=2, min_value=1)
    with col2: y_col_idx = st.number_input("縦軸 (I/A) 列", value=3, min_value=1)
    skip_rows = st.number_input("ヘッダー行数", value=1, min_value=0)
    data_sep = st.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)
    smoothing = st.checkbox("スムージング (ノイズ除去)", value=True)

# 2. サンプルファイルアップロード (共通化)
st.sidebar.markdown("---")
st.sidebar.subheader("📥 サンプルデータのアップロード")
sample_files = st.sidebar.file_uploader(
    "解析・比較したいファイルを全て選択", 
    type=['csv', 'txt', 'dat'], 
    accept_multiple_files=True, 
    key="sample_upload_sidebar"
)

# 3. グラフ表示設定
st.sidebar.markdown("---")
with st.sidebar.expander("📊 グラフ表示設定", expanded=False):
    line_width = st.slider("線の太さ", 0.5, 5.0, 2.0, 0.1)
    color_mode = st.radio("配色モード", ["自動 (複数色)", "単色指定"], horizontal=True)
    custom_color = st.color_picker("単色時の色", "#000000")
    show_grid = st.checkbox("グリッド線", value=True)
    show_ticks = st.checkbox("目盛ラベル", value=True)
    show_mirror = st.checkbox("枠線 (Mirror Axis)", value=True)
    font_size = st.number_input("フォントサイズ", value=14, min_value=8)

# ==========================================
# タブ構成
# ==========================================
# 新しいタブ構成
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ 校正", 
    "2️⃣ 個別解析", 
    "3️⃣ 比較・重ね書き", 
    "📝 HOMO/LUMO", 
    "ℹ️ メモ・原理"
])

# ==========================================
# Tab 1: 校正
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")
    st.caption("ここで決定したシフト値は、他のすべてのタブに適用されます。")
    
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
                    st.success(f"校正完了！ シフト値: {E_half:.4f} V を保存しました。")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=volt, y=curr, mode='lines', name='Raw', line=dict(color=custom_color, width=line_width)))
                fig.add_trace(go.Scatter(x=[E_pa], y=[I_pa], mode='markers', name='Anodic', marker=dict(color='red', size=10)))
                fig.add_trace(go.Scatter(x=[E_pc], y=[I_pc], mode='markers', name='Cathodic', marker=dict(color='blue', size=10)))
                fig.add_vline(x=E_half, line_dash="dash", line_color="green", annotation_text="E 1/2")
                fig = update_fig_layout(fig, f"Standard ({fc_file.name})", "V", "A", show_grid, show_mirror, show_ticks, font_size)
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Tab 2: 個別解析 (ピーク登録)
# ==========================================
with tab2:
    st.header("サンプルデータの解析")
    
    shift_val = st.session_state['calibration_shift']
    if st.session_state['is_calibrated']:
        st.success(f"✅ 現在の補正値: **{shift_val:.4f} V**")
    else:
        st.warning("⚠️ 未校正 (元の電圧表示)")

    if sample_files:
        # データ準備
        data_cache = {}
        for s_file in sample_files:
            df_s = load_data(s_file, skip_rows, sep=data_sep)
            if df_s is not None and df_s.shape[1] >= max(x_col_idx, y_col_idx):
                v_raw = df_s.iloc[:, x_col_idx - 1].values
                i_raw = df_s.iloc[:, y_col_idx - 1].values
                if smoothing: i_raw = smooth_data(i_raw)
                data_cache[s_file.name] = {"v": v_raw - shift_val, "i": i_raw}

        # ピーク解析UI
        st.subheader("ピーク検出・登録")
        target_name = st.selectbox("解析するファイルを選択", list(data_cache.keys()), key="tab2_select")
        
        if target_name:
            v_tgt = data_cache[target_name]["v"]
            i_tgt = data_cache[target_name]["i"]
            
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                st.markdown("**① 探索範囲**")
                p_min = st.number_input("Min (V)", value=float(np.min(v_tgt)), step=0.1, format="%.2f", key="p_min")
                p_max = st.number_input("Max (V)", value=float(np.max(v_tgt)), step=0.1, format="%.2f", key="p_max")
                
                mask_s = (v_tgt >= p_min) & (v_tgt <= p_max)
                if any(mask_s):
                    auto_epa = v_tgt[mask_s][np.argmax(i_tgt[mask_s])]
                    auto_epc = v_tgt[mask_s][np.argmin(i_tgt[mask_s])]
                else:
                    auto_epa, auto_epc = 0.0, 0.0

            with col_in2:
                st.markdown("**② 登録**")
                m_epa = st.number_input("Epa (V)", value=float(auto_epa), format="%.4f", key="m_epa")
                m_epc = st.number_input("Epc (V)", value=float(auto_epc), format="%.4f", key="m_epc")
                m_half = (m_epa + m_epc) / 2
                st.caption(f"Calculated E1/2: {m_half:.4f} V")
                
                if st.button("リストに追加 ✅", key="add_peak"):
                    st.session_state['peak_results'].append({
                        "File": target_name, "E_pa": m_epa, "E_pc": m_epc, "E_1/2": m_half
                    })
                    st.success("Added!")

            # 結果表示
            if st.session_state['peak_results']:
                st.dataframe(pd.DataFrame(st.session_state['peak_results']), use_container_width=True)
                if st.button("クリア 🗑️"):
                    st.session_state['peak_results'] = []
                    st.rerun()

            # 解析用グラフ
            fig_check = go.Figure()
            fig_check.add_trace(go.Scatter(x=v_tgt, y=i_tgt, mode='lines', line=dict(color='black')))
            fig_check.add_trace(go.Scatter(x=v_tgt[mask_s], y=i_tgt[mask_s], mode='lines', line=dict(color='orange', width=3), opacity=0.5, name="Search Range"))
            
            # 登録済みピークプロット
            for p in [x for x in st.session_state['peak_results'] if x['File'] == target_name]:
                fig_check.add_vline(x=p["E_1/2"], line_dash="dot", line_color="green")
                fig_check.add_trace(go.Scatter(x=[p["E_pa"]], y=[np.max(i_tgt)], mode='markers', marker=dict(symbol='star', size=10, color='red'), showlegend=False))
            
            fig_check = update_fig_layout(fig_check, f"Analysis: {target_name}", "V vs Fc/Fc+", "I", show_grid, show_mirror, show_ticks, font_size)
            st.plotly_chart(fig_check, use_container_width=True)
    else:
        st.info("👈 サイドバーからサンプルデータをアップロードしてください。")

# ==========================================
# Tab 3: 比較・重ね書き (新規追加)
# ==========================================
with tab3:
    st.header("📊 データの比較・重ね書き")
    st.markdown("アップロードされたデータから選択して、重ね書きや比較作図を行います。")

    if sample_files:
        # データ読み込み (キャッシュ)
        data_cache_comp = {}
        for s_file in sample_files:
            df_s = load_data(s_file, skip_rows, sep=data_sep)
            if df_s is not None and df_s.shape[1] >= max(x_col_idx, y_col_idx):
                v_raw = df_s.iloc[:, x_col_idx - 1].values
                i_raw = df_s.iloc[:, y_col_idx - 1].values
                if smoothing: i_raw = smooth_data(i_raw)
                data_cache_comp[s_file.name] = {"v": v_raw - st.session_state['calibration_shift'], "i": i_raw}

        # --- 作図設定 ---
        col_opt1, col_opt2 = st.columns([1, 2])
        
        with col_opt1:
            st.subheader("設定")
            # ファイル選択
            selected_files = st.multiselect(
                "表示するファイルを選択", 
                options=list(data_cache_comp.keys()), 
                default=list(data_cache_comp.keys())
            )
            
            # オプション
            st.markdown("**調整オプション**")
            normalize = st.checkbox("最大値で正規化 (Normalize)", value=False)
            y_offset = st.number_input("Y軸オフセット (ずらして表示)", value=0.0, step=1e-6, format="%.2e")
            
        with col_opt2:
            st.subheader("プレビュー")
            if selected_files:
                fig_comp = go.Figure()
                colors = pc.qualitative.Plotly

                for idx, fname in enumerate(selected_files):
                    v_dat = data_cache_comp[fname]["v"]
                    i_dat = data_cache_comp[fname]["i"]
                    
                    # 正規化処理
                    if normalize:
                        max_val = np.max(np.abs(i_dat))
                        if max_val > 0:
                            i_dat = i_dat / max_val
                    
                    # オフセット処理
                    i_dat = i_dat + (y_offset * idx)

                    # 色決定
                    line_c = colors[idx % len(colors)] if color_mode == "自動 (複数色)" else custom_color
                    
                    fig_comp.add_trace(go.Scatter(
                        x=v_dat, y=i_dat, 
                        mode='lines', 
                        name=fname,
                        line=dict(color=line_c, width=line_width)
                    ))

                # 軸ラベルの動的変更
                y_label = "Normalized Current / a.u." if normalize else "Current / A"
                
                fig_comp = update_fig_layout(fig_comp, "Comparison Plot", "Potential vs Fc/Fc+ / V", y_label, show_grid, show_mirror, show_ticks, font_size)
                fig_comp.add_vline(x=0, line_color="gray", line_width=1)
                fig_comp.add_hline(y=0, line_color="gray", line_width=1)
                
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("表示するファイルを選択してください。")
    else:
        st.info("👈 サイドバーからデータをアップロードしてください。")

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