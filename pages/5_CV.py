import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.signal import find_peaks, savgol_filter

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro Custom", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer Pro")
st.markdown("標準物質による校正、任意の列指定、**グラフのカスタマイズ**が可能です。")

# --- セッション状態の初期化 ---
if 'calibration_shift' not in st.session_state:
    st.session_state['calibration_shift'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False

# --- 関数定義 ---
def load_data(uploaded_file, skip_rows, encoding='utf-8', sep='auto'):
    """データの読み込み関数"""
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
    except Exception as e:
        return None

def smooth_data(y, window_length=11, polyorder=3):
    """平滑化処理 (Savitzky-Golay filter)"""
    try:
        if window_length > len(y):
            window_length = len(y) // 2 * 2 + 1 
        return savgol_filter(y, window_length, polyorder)
    except:
        return y

# ==========================================
# サイドバー設定
# ==========================================
st.sidebar.header("📂 データ読み込み設定")
col1, col2 = st.sidebar.columns(2)
with col1:
    x_col_idx = st.number_input("横軸 (E/V) 列番号", value=2, min_value=1)
with col2:
    y_col_idx = st.number_input("縦軸 (I/A) 列番号", value=3, min_value=1)

skip_rows = st.sidebar.number_input("ヘッダー行数", value=1, min_value=0)
data_sep = st.sidebar.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)
smoothing = st.sidebar.checkbox("スムージング (ノイズ除去)", value=True)

st.sidebar.markdown("---")

# --- グラフ表示設定 (新規追加) ---
with st.sidebar.expander("📊 グラフ表示設定", expanded=True):
    st.markdown("**スタイル**")
    line_width = st.slider("線の太さ", 0.5, 5.0, 2.0, 0.1)
    
    # 色設定
    color_mode = st.radio("サンプル(Tab2)の色設定", ["自動 (複数色)", "単色指定"], horizontal=True)
    custom_color = st.color_picker("プロットの色 (単色指定時)", "#000000")
    
    st.markdown("**軸・グリッド**")
    show_grid = st.checkbox("グリッド線を表示", value=True)
    show_ticks = st.checkbox("目盛ラベルを表示", value=True)
    show_mirror = st.checkbox("枠線 (Mirror Axis) を表示", value=True)
    
    font_size = st.number_input("フォントサイズ", value=14, min_value=8, max_value=30)

# --- 共通のレイアウト更新関数 ---
def update_fig_layout(fig, title, x_title, y_title):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=600,
        font=dict(size=font_size),
        hovermode="closest",
        template="simple_white", # ベースはシンプルに
        xaxis=dict(
            showgrid=show_grid, 
            gridcolor='lightgray',
            showline=True, 
            mirror=show_mirror, 
            ticks="outside" if show_ticks else "",
            showticklabels=show_ticks
        ),
        yaxis=dict(
            showgrid=show_grid, 
            gridcolor='lightgray',
            showline=True, 
            mirror=show_mirror,
            ticks="outside" if show_ticks else "",
            showticklabels=show_ticks
        )
    )
    return fig

# --- タブ構成 ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 標準物質 (Ferrocene) 校正", "2️⃣ サンプル解析", "📝 HOMO/LUMO 計算"])

# ==========================================
# Tab 1: フェロセンによる校正
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")
    
    fc_file = st.file_uploader("フェロセン (標準物質) のデータ", type=['csv', 'txt', 'dat'], key="fc_upload")

    if fc_file:
        df_fc = load_data(fc_file, skip_rows, sep=data_sep)
        max_col = max(x_col_idx, y_col_idx)
        
        if df_fc is not None and df_fc.shape[1] >= max_col:
            volt = df_fc.iloc[:, x_col_idx - 1].values
            curr = df_fc.iloc[:, y_col_idx - 1].values
            
            if smoothing:
                curr = smooth_data(curr)

            # ピーク検出
            st.subheader("ピーク検出設定")
            col_range1, col_range2 = st.columns(2)
            min_v, max_v = float(np.min(volt)), float(np.max(volt))
            with col_range1:
                search_min = st.number_input("探索範囲 Min (V)", value=min_v, format="%.2f")
            with col_range2:
                search_max = st.number_input("探索範囲 Max (V)", value=max_v, format="%.2f")

            mask = (volt >= search_min) & (volt <= search_max)
            v_roi = volt[mask]
            c_roi = curr[mask]

            if len(v_roi) > 0:
                idx_max = np.argmax(c_roi)
                E_pa = v_roi[idx_max]
                I_pa = c_roi[idx_max]
                idx_min = np.argmin(c_roi)
                E_pc = v_roi[idx_min]
                I_pc = c_roi[idx_min]
                E_half = (E_pa + E_pc) / 2

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("酸化ピーク $E_{pa}$", f"{E_pa:.3f} V")
                col_res2.metric("還元ピーク $E_{pc}$", f"{E_pc:.3f} V")
                col_res3.metric("式量電位 $E_{1/2}$", f"{E_half:.3f} V")

                if st.button("この値を基準 (0 V) に設定する"):
                    st.session_state['calibration_shift'] = E_half
                    st.session_state['is_calibrated'] = True
                    st.success(f"校正完了！ シフト値: {E_half:.4f} V を保存しました。")

                # --- グラフ描画 (Tab1) ---
                fig = go.Figure()
                
                # 色設定: Tab1は常に指定色を使用
                plot_color = custom_color
                
                fig.add_trace(go.Scatter(
                    x=volt, y=curr, 
                    mode='lines', 
                    name='Raw Data', 
                    line=dict(color=plot_color, width=line_width)
                ))
                # ピーク
                fig.add_trace(go.Scatter(x=[E_pa], y=[I_pa], mode='markers', name='Anodic Peak', marker=dict(color='red', size=10)))
                fig.add_trace(go.Scatter(x=[E_pc], y=[I_pc], mode='markers', name='Cathodic Peak', marker=dict(color='blue', size=10)))
                # E1/2 線
                fig.add_vline(x=E_half, line_dash="dash", line_color="green", annotation_text="E 1/2")

                fig = update_fig_layout(fig, f"Standard Substance ({fc_file.name})", "Potential / V", "Current / A")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("指定された範囲内にデータがありません。")
        else:
            st.error("データの読み込みに失敗しました。")

# ==========================================
# Tab 2: サンプル解析
# ==========================================
with tab2:
    st.header("サンプルデータの解析 (校正済み)")
    
    shift_val = st.session_state['calibration_shift']
    if st.session_state['is_calibrated']:
        st.success(f"✅ 現在の補正値: **{shift_val:.4f} V** (この値が引かれます)")
    else:
        st.warning("⚠️ まだ校正が行われていません。元の電圧がそのまま表示されます。")

    sample_files = st.file_uploader("サンプルデータ (複数可)", type=['csv', 'txt', 'dat'], accept_multiple_files=True, key="sample_upload")

    if sample_files:
        st.subheader("補正後のCVプロット")
        show_raw = st.checkbox("補正前のデータも重ねて表示する", value=False)
        
        fig_sample = go.Figure()
        
        # 色サイクル (自動モード用)
        colors = pc.qualitative.Plotly
        
        for idx, s_file in enumerate(sample_files):
            df_s = load_data(s_file, skip_rows, sep=data_sep)
            max_col = max(x_col_idx, y_col_idx)
            
            if df_s is not None and df_s.shape[1] >= max_col:
                v_raw = df_s.iloc[:, x_col_idx - 1].values
                i_raw = df_s.iloc[:, y_col_idx - 1].values
                
                if smoothing:
                    i_raw = smooth_data(i_raw)

                v_calibrated = v_raw - shift_val
                
                # 色の決定
                if color_mode == "自動 (複数色)":
                    line_c = colors[idx % len(colors)]
                else:
                    line_c = custom_color

                # 補正後プロット
                fig_sample.add_trace(go.Scatter(
                    x=v_calibrated, y=i_raw, 
                    mode='lines', 
                    name=f"{s_file.name}",
                    line=dict(color=line_c, width=line_width)
                ))

                # 補正前プロット (Raw) - 薄く表示
                if show_raw:
                    fig_sample.add_trace(go.Scatter(
                        x=v_raw, y=i_raw,
                        mode='lines',
                        name=f"{s_file.name} (Raw)",
                        line=dict(dash='dash', width=max(1.0, line_width-1), color='darkgray'),
                        opacity=0.6,
                        showlegend=False
                    ))
            else:
                st.warning(f"{s_file.name}: 列不足のためスキップ")

        # 軸設定の適用
        fig_sample = update_fig_layout(fig_sample, "Sample CV (vs Fc/Fc+)", "Potential vs Fc/Fc+ / V", "Current / A")
        
        # 0点ライン
        fig_sample.add_vline(x=0, line_color="gray", line_width=1)
        fig_sample.add_hline(y=0, line_color="gray", line_width=1)
        
        st.plotly_chart(fig_sample, use_container_width=True)

# ==========================================
# Tab 3: HOMO/LUMO 計算
# ==========================================
with tab3:
    st.header("🧪 HOMO / LUMO レベルの算出")
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        st.subheader("HOMO 計算")
        e_onset_ox = st.number_input("酸化開始電位 (vs Fc/Fc+) [V]", value=0.5, step=0.01)
        fc_level = st.number_input("フェロセンの基準エネルギー [eV]", value=4.8, step=0.1)
        homo = -(e_onset_ox + fc_level)
        st.metric("HOMO Level", f"{homo:.2f} eV")
    with col_calc2:
        st.subheader("LUMO 計算")
        e_onset_red = st.number_input("還元開始電位 (vs Fc/Fc+) [V]", value=-1.5, step=0.01)
        lumo = -(e_onset_red + fc_level)
        st.metric("LUMO Level", f"{lumo:.2f} eV")