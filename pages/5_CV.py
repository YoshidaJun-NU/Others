import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.signal import savgol_filter

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro Custom", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer Pro")
st.markdown("標準物質による校正、**複数ピークの検出と手動入力**に対応しています。")

# --- セッション状態の初期化 ---
if 'calibration_shift' not in st.session_state:
    st.session_state['calibration_shift'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False
# 保存されたピークリストの初期化
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
        height=500, font=dict(size=font_size), hovermode="closest", template="simple_white",
        xaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, mirror=show_mirror, ticks="outside" if show_ticks else "", showticklabels=show_ticks),
        yaxis=dict(showgrid=show_grid, gridcolor='lightgray', showline=True, mirror=show_mirror, ticks="outside" if show_ticks else "", showticklabels=show_ticks)
    )
    return fig

# --- サイドバー設定 ---
st.sidebar.header("📂 データ読み込み設定")
col1, col2 = st.sidebar.columns(2)
with col1: x_col_idx = st.number_input("横軸 (E/V) 列番号", value=2, min_value=1)
with col2: y_col_idx = st.number_input("縦軸 (I/A) 列番号", value=3, min_value=1)

skip_rows = st.sidebar.number_input("ヘッダー行数", value=1, min_value=0)
data_sep = st.sidebar.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)
smoothing = st.sidebar.checkbox("スムージング (ノイズ除去)", value=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📊 グラフ表示設定", expanded=False):
    line_width = st.slider("線の太さ", 0.5, 5.0, 2.0, 0.1)
    color_mode = st.radio("サンプル色設定", ["自動 (複数色)", "単色指定"], horizontal=True)
    custom_color = st.color_picker("プロットの色", "#000000")
    show_grid = st.checkbox("グリッド線", value=True)
    show_ticks = st.checkbox("目盛ラベル", value=True)
    show_mirror = st.checkbox("枠線 (Mirror Axis)", value=True)
    font_size = st.number_input("フォントサイズ", value=14, min_value=8)

# --- タブ構成 ---
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 標準物質 (校正)", "2️⃣ サンプル解析 (複数ピーク)", "📝 HOMO/LUMO 計算", "📝 メモ・原理"])

# ==========================================
# Tab 1: 校正
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")
    fc_file = st.file_uploader("標準物質データ", type=['csv', 'txt', 'dat'], key="fc_upload")

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
# Tab 2: サンプル解析 (マルチピーク & 手動入力)
# ==========================================
with tab2:
    st.header("サンプルデータの解析")
    
    shift_val = st.session_state['calibration_shift']
    if st.session_state['is_calibrated']:
        st.success(f"✅ 現在の補正値: **{shift_val:.4f} V**")
    else:
        st.warning("⚠️ 未校正 (元の電圧表示)")

    sample_files = st.file_uploader("サンプルデータ (複数可)", type=['csv', 'txt', 'dat'], accept_multiple_files=True, key="sample_upload")

    if sample_files:
        # --- データ読み込みと全体プロット ---
        st.subheader("1. 全体プロット")
        fig_sample = go.Figure()
        colors = pc.qualitative.Plotly
        data_cache = {}

        for idx, s_file in enumerate(sample_files):
            df_s = load_data(s_file, skip_rows, sep=data_sep)
            max_col = max(x_col_idx, y_col_idx)
            if df_s is not None and df_s.shape[1] >= max_col:
                v_raw = df_s.iloc[:, x_col_idx - 1].values
                i_raw = df_s.iloc[:, y_col_idx - 1].values
                if smoothing: i_raw = smooth_data(i_raw)
                v_calibrated = v_raw - shift_val
                
                data_cache[s_file.name] = {"v": v_calibrated, "i": i_raw}
                
                line_c = colors[idx % len(colors)] if color_mode == "自動 (複数色)" else custom_color
                fig_sample.add_trace(go.Scatter(
                    x=v_calibrated, y=i_raw, mode='lines', name=f"{s_file.name}",
                    line=dict(color=line_c, width=line_width)
                ))

        fig_sample = update_fig_layout(fig_sample, "All Samples", "V vs Fc/Fc+", "Current / A", show_grid, show_mirror, show_ticks, font_size)
        st.plotly_chart(fig_sample, use_container_width=True)

        # --- 詳細解析セクション ---
        st.divider()
        st.subheader("2. 詳細解析 (ピーク登録)")
        st.info("範囲を指定して自動検出するか、値を直接入力して「リストに追加」してください。複数のピークを記録できます。")

        target_name = st.selectbox("解析するファイルを選択", list(data_cache.keys()))
        
        if target_name:
            v_tgt = data_cache[target_name]["v"]
            i_tgt = data_cache[target_name]["i"]
            
            # --- 範囲指定と自動検出 ---
            col_in1, col_in2 = st.columns(2)
            s_min_def, s_max_def = float(np.min(v_tgt)), float(np.max(v_tgt))
            
            with col_in1:
                st.markdown("**① 探索範囲の設定**")
                peak_min = st.number_input("Min (V)", value=s_min_def, step=0.1, format="%.2f")
                peak_max = st.number_input("Max (V)", value=s_max_def, step=0.1, format="%.2f")
            
            # 範囲内データ抽出と自動検出
            mask_s = (v_tgt >= peak_min) & (v_tgt <= peak_max)
            if any(mask_s):
                auto_epa = v_tgt[mask_s][np.argmax(i_tgt[mask_s])]
                auto_epc = v_tgt[mask_s][np.argmin(i_tgt[mask_s])]
            else:
                auto_epa, auto_epc = 0.0, 0.0

            with col_in2:
                st.markdown("**② 値の確認と修正 (手動入力可)**")
                # 手動入力（デフォルトは自動検出値）
                manual_epa = st.number_input("酸化ピーク Epa (V)", value=float(auto_epa), format="%.4f", step=0.01)
                manual_epc = st.number_input("還元ピーク Epc (V)", value=float(auto_epc), format="%.4f", step=0.01)
                
                # 計算
                manual_ehalf = (manual_epa + manual_epc) / 2
                st.markdown(f"計算された $E_{{1/2}}$: **{manual_ehalf:.4f} V**")
                
                # 追加ボタン
                if st.button("リストに追加 ✅"):
                    st.session_state['peak_results'].append({
                        "File": target_name,
                        "E_pa (V)": manual_epa,
                        "E_pc (V)": manual_epc,
                        "E_1/2 (V)": manual_ehalf
                    })
                    st.success("追加しました！")

            # --- 結果リストの表示 ---
            if len(st.session_state['peak_results']) > 0:
                st.markdown("### 📋 登録済みピークリスト")
                
                # 現在のファイルに関連するものだけ表示するか、全表示するか
                res_df = pd.DataFrame(st.session_state['peak_results'])
                st.dataframe(res_df, use_container_width=True)
                
                if st.button("リストをクリア 🗑️"):
                    st.session_state['peak_results'] = []
                    st.rerun()

            # --- 確認用グラフ (登録済みピークを重ねて表示) ---
            st.markdown("### 📈 確認用プロット")
            fig_check = go.Figure()
            # 全データ
            fig_check.add_trace(go.Scatter(x=v_tgt, y=i_tgt, mode='lines', name='Full Data', line=dict(color='black', width=2)))
            # 選択範囲 (視覚化)
            mask_roi = (v_tgt >= peak_min) & (v_tgt <= peak_max)
            fig_check.add_trace(go.Scatter(x=v_tgt[mask_roi], y=i_tgt[mask_roi], mode='lines', name='Selected Range', line=dict(color='orange', width=4), opacity=0.5))
            
            # リストにあるピークをプロット
            current_file_peaks = [p for p in st.session_state['peak_results'] if p['File'] == target_name]
            for i, p in enumerate(current_file_peaks):
                fig_check.add_trace(go.Scatter(
                    x=[p["E_pa (V)"]], y=[np.max(i_tgt)], # Y位置は便宜上Maxに合わせるか、厳密にはIが必要だが簡易表示
                    mode='markers+text', text=[f"P{i+1}"], textposition="top center",
                    marker=dict(color='red', size=12, symbol="star"), name=f"Saved Peak {i+1} (ox)"
                ))
                fig_check.add_trace(go.Scatter(
                    x=[p["E_pc (V)"]], y=[np.min(i_tgt)],
                    mode='markers', marker=dict(color='blue', size=12, symbol="star"), name=f"Saved Peak {i+1} (red)"
                ))
                # E1/2 ライン
                fig_check.add_vline(x=p["E_1/2 (V)"], line_dash="dot", line_color="green", opacity=0.5)

            fig_check = update_fig_layout(fig_check, f"Analysis: {target_name}", "V vs Fc/Fc+", "Current / A", show_grid, show_mirror, show_ticks, font_size)
            st.plotly_chart(fig_check, use_container_width=True)

# ==========================================
# Tab 3: HOMO/LUMO 計算
# ==========================================
with tab3:
    st.header("🧪 HOMO / LUMO レベルの算出")
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        st.subheader("HOMO 計算")
        e_onset_ox = st.number_input("酸化開始電位 (vs Fc/Fc+) [V]", value=0.5, step=0.01)
        fc_level = st.number_input("フェロセンの基準 [eV]", value=4.8, step=0.1)
        homo = -(e_onset_ox + fc_level)
        st.metric("HOMO Level", f"{homo:.2f} eV")
    with col_calc2:
        st.subheader("LUMO 計算")
        e_onset_red = st.number_input("還元開始電位 (vs Fc/Fc+) [V]", value=-1.5, step=0.01)
        lumo = -(e_onset_red + fc_level)
        st.metric("LUMO Level", f"{lumo:.2f} eV")

# ==========================================
# Tab 4: メモ・原理
# ==========================================
with tab4:
    st.header("📝 メモ・原理")
    st.markdown(EXPLANATION_TEXT)