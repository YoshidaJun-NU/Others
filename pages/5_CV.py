import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer Pro")
st.markdown("""
標準物質（フェロセン等）による電位校正と、任意フォーマットのデータ解析が可能です。
サイドバーで **「使用する列」** を指定してください。
""")

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
        # 区切り文字の自動判定ロジック
        if sep == 'auto':
            # まずcsvとして読んでみる
            try:
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, engine='python', encoding=encoding)
                if df.shape[1] <= 1:
                    # 1列にしかならない場合はスペース/タブ区切りを試す
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python', encoding=encoding)
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=r'\s+', engine='python', encoding=encoding)
        else:
            df = pd.read_csv(uploaded_file, skiprows=skip_rows, header=None, sep=sep, engine='python', encoding=encoding)
        
        # 数値変換（変換できない文字が含まれる行は削除）
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except Exception as e:
        return None

def smooth_data(y, window_length=11, polyorder=3):
    """平滑化処理 (Savitzky-Golay filter)"""
    try:
        if window_length > len(y):
            window_length = len(y) // 2 * 2 + 1 # 奇数にする
        return savgol_filter(y, window_length, polyorder)
    except:
        return y

# --- サイドバー：読み込み設定 ---
st.sidebar.header("📂 データ読み込み設定")

# 列指定（ユーザーの要望に対応）
col1, col2 = st.sidebar.columns(2)
with col1:
    x_col_idx = st.number_input("横軸 (E/V) の列番号", value=2, min_value=1, help="データの何列目をX軸として使うか指定します（1から開始）")
with col2:
    y_col_idx = st.number_input("縦軸 (I/A) の列番号", value=3, min_value=1, help="データの何列目をY軸として使うか指定します（1から開始）")

# その他の設定
skip_rows = st.sidebar.number_input("ヘッダー行数 (スキップ)", value=1, min_value=0, help="データの先頭にあるラベル行などを飛ばす行数。添付ファイルの場合は「1」推奨。")
data_sep = st.sidebar.selectbox("区切り文字", ['auto', ',', '\t', ' '], index=0)
smoothing = st.sidebar.checkbox("スムージング (ノイズ除去)", value=True)

# --- タブ構成 ---
tab1, tab2, tab3 = st.tabs(["1️⃣ 標準物質 (Ferrocene) 校正", "2️⃣ サンプル解析", "📝 HOMO/LUMO 計算"])

# ==========================================
# Tab 1: フェロセンによる校正
# ==========================================
with tab1:
    st.header("標準物質による基準電位の決定")
    st.info("標準物質 (例: 231130_ferrocene.txt) をアップロードし、E1/2 を決定してください。")

    fc_file = st.file_uploader("フェロセン (標準物質) のデータ", type=['csv', 'txt', 'dat'], key="fc_upload")

    if fc_file:
        # データの読み込み
        df_fc = load_data(fc_file, skip_rows, sep=data_sep)
        
        # 指定された列が存在するかチェック
        max_col = max(x_col_idx, y_col_idx)
        
        if df_fc is not None and df_fc.shape[1] >= max_col:
            # ユーザー指定の列を取得 (内部indexは0始まりなので -1 する)
            volt = df_fc.iloc[:, x_col_idx - 1].values
            curr = df_fc.iloc[:, y_col_idx - 1].values
            
            if smoothing:
                curr = smooth_data(curr)

            # --- ピーク検出 ---
            st.subheader("ピーク検出設定")
            col_range1, col_range2 = st.columns(2)
            min_v, max_v = float(np.min(volt)), float(np.max(volt))
            
            with col_range1:
                search_min = st.number_input("探索範囲 Min (V)", value=min_v, format="%.2f")
            with col_range2:
                search_max = st.number_input("探索範囲 Max (V)", value=max_v, format="%.2f")

            # 範囲内データ抽出
            mask = (volt >= search_min) & (volt <= search_max)
            v_roi = volt[mask]
            c_roi = curr[mask]

            if len(v_roi) > 0:
                # 酸化ピーク (Current Max)
                idx_max = np.argmax(c_roi)
                E_pa = v_roi[idx_max]
                I_pa = c_roi[idx_max]

                # 還元ピーク (Current Min)
                idx_min = np.argmin(c_roi)
                E_pc = v_roi[idx_min]
                I_pc = c_roi[idx_min]

                # E_1/2
                E_half = (E_pa + E_pc) / 2

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("酸化ピーク $E_{pa}$", f"{E_pa:.3f} V")
                col_res2.metric("還元ピーク $E_{pc}$", f"{E_pc:.3f} V")
                col_res3.metric("式量電位 $E_{1/2}$", f"{E_half:.3f} V")

                if st.button("この値を基準 (0 V) に設定する"):
                    st.session_state['calibration_shift'] = E_half
                    st.session_state['is_calibrated'] = True
                    st.success(f"校正完了！ シフト値: {E_half:.4f} V を保存しました。")

                # グラフ
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=volt, y=curr, mode='lines', name='Raw Data', line=dict(color='black')))
                fig.add_trace(go.Scatter(x=[E_pa], y=[I_pa], mode='markers', name='Anodic Peak', marker=dict(color='red', size=10)))
                fig.add_trace(go.Scatter(x=[E_pc], y=[I_pc], mode='markers', name='Cathodic Peak', marker=dict(color='blue', size=10)))
                fig.add_vline(x=E_half, line_dash="dash", line_color="green", annotation_text="E 1/2")

                fig.update_layout(
                    title=f"Standard Substance ({fc_file.name})",
                    xaxis_title="Potential / V",
                    yaxis_title="Current / A",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("指定された範囲内にデータがありません。")

        elif df_fc is not None:
            st.error(f"指定された列番号（{max_col}列目）がデータ内に存在しません。データは全{df_fc.shape[1]}列です。")
            st.dataframe(df_fc.head()) # デバッグ用に先頭を表示
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
        show_raw = st.checkbox("補正前のデータも点線で表示", value=False)
        
        fig_sample = go.Figure()

        for s_file in sample_files:
            df_s = load_data(s_file, skip_rows, sep=data_sep)
            
            # 列チェック
            max_col = max(x_col_idx, y_col_idx)
            if df_s is not None and df_s.shape[1] >= max_col:
                v_raw = df_s.iloc[:, x_col_idx - 1].values
                i_raw = df_s.iloc[:, y_col_idx - 1].values
                
                if smoothing:
                    i_raw = smooth_data(i_raw)

                # 電圧補正
                v_calibrated = v_raw - shift_val

                fig_sample.add_trace(go.Scatter(
                    x=v_calibrated, y=i_raw, 
                    mode='lines', 
                    name=f"{s_file.name} (Calibrated)",
                    line=dict(width=2)
                ))

                if show_raw:
                    fig_sample.add_trace(go.Scatter(
                        x=v_raw, y=i_raw,
                        mode='lines',
                        name=f"{s_file.name} (Raw)",
                        line=dict(dash='dot', width=1),
                        visible='legendonly'
                    ))
            else:
                st.warning(f"{s_file.name}: 指定された列が存在しません。")

        fig_sample.update_layout(
            title="Sample CV (vs Fc/Fc+)",
            xaxis_title="Potential vs Fc/Fc+ / V",
            yaxis_title="Current / A",
            height=600,
            hovermode="closest"
        )
        fig_sample.add_vline(x=0, line_color="gray", line_width=1)
        fig_sample.add_hline(y=0, line_color="gray", line_width=1)
        
        st.plotly_chart(fig_sample, use_container_width=True)

# ==========================================
# Tab 3: HOMO/LUMO 計算
# ==========================================
with tab3:
    st.header("🧪 HOMO / LUMO レベルの算出")
    st.markdown("""
    校正されたCVの酸化開始電位 ($E_{onset, ox}$) や還元開始電位 ($E_{onset, red}$) からエネルギー準位を計算します。
    """)

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