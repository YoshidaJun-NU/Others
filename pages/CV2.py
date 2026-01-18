import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from scipy.signal import savgol_filter, find_peaks

# --- ページ設定 ---
st.set_page_config(page_title="CV Analyzer Pro (Multi-Cycle)", layout="wide")
st.title("⚡ Cyclic Voltammetry Analyzer Pro")
st.markdown("標準物質校正、**サイクル別解析**、**複数ピーク検出**に対応した高機能版です。")

# --- セッション状態の初期化 ---
if 'calibration_shift' not in st.session_state:
    st.session_state['calibration_shift'] = 0.0
if 'is_calibrated' not in st.session_state:
    st.session_state['is_calibrated'] = False
if 'peak_results' not in st.session_state:
    st.session_state['peak_results'] = []

# --- 解説テキスト ---
EXPLANATION_TEXT = """
### 📚 サイクル分割とピーク検出について

#### 1. サイクル分割 (Cycle Splitting)
連続して測定された複数回のスキャンデータ（マルチサイクル）を、個別のサイクルに分割して解析できます。
* **初期電圧・最大電圧・最小電圧**を入力することで、電圧の折り返し点を自動検出し、サイクルを切り分けます。
* 分割されたデータを選択すると、そのサイクルだけのピーク解析が可能になります。

#### 2. 複数ピーク検出
指定した範囲内に存在する**複数の酸化ピーク（極大）**と**還元ピーク（極小）**を自動で探します。
* **Prominence (突出度)**: 周囲のベースラインからどれくらい飛び出しているかを基準に検出します。ノイズを拾う場合はこの値を大きくしてください。
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
    """
    指定範囲内のデータから複数のピークを検索してリストで返す。
    """
    # データの振幅
    amplitude = np.max(i) - np.min(i)
    prom = amplitude * prominence_val if amplitude > 0 else None

    # 上に凸 (Maxima)
    peaks_top_idx, _ = find_peaks(i, prominence=prom)
    peaks_top = [{"E": v[idx], "I": i[idx], "Type": "Anodic (Top)"} for idx in peaks_top_idx]

    # 下に凸 (Minima) -> -i に対して検索
    peaks_btm_idx, _ = find_peaks(-i, prominence=prom)
    peaks_btm = [{"E": v[idx], "I": i[idx], "Type": "Cathodic (Bottom)"} for idx in peaks_btm_idx]

    # Eの値順（電圧が低い順）にソート
    peaks_top.sort(key=lambda x: x["E"])
    peaks_btm.sort(key=lambda x: x["E"])

    return peaks_top, peaks_btm

def split_cycles_by_voltage(v, i, v_init, v_max, v_min):
    """
    電圧の折り返し点に基づいてサイクルを分割する簡易ロジック
    """
    # 単純化のため、v_initに戻った回数で区切る、あるいは極値のペアで区切る
    # ここでは「初期値付近を通過」かつ「傾きが開始時と同じ」で分割点を推定
    
    # 1. データの微分（方向）
    grad = np.gradient(v)
    start_sign = np.sign(grad[0]) if abs(grad[0]) > 0 else 1 # 開始時のスイープ方向

    # 2. v_init との交差判定 (初期値から許容誤差範囲内 かつ 方向が一致)
    # 許容誤差: スイープ幅の 1%
    tol = (max(v) - min(v)) * 0.05
    
    # 候補点を探す
    # Init付近 かつ 傾き方向が一致するインデックス
    candidates = []
    
    # ノイズ対策として少しスキップしてから探索開始
    min_points_per_cycle = 10 
    
    last_idx = 0
    cycles = []
    
    # データ全体を走査して分割点を探すのは複雑なので、
    # 簡易的に「極大・極小のセット」を1サイクルとみなすアプローチをとる
    
    # 極大点(High)と極小点(Low)のインデックスを探す
    # ユーザー入力のV_max, V_minに近い点を探す
    
    # 全体の極大・極小候補
    peaks_high, _ = find_peaks(v, height=v_max - abs(v_max)*0.1) # Max付近
    peaks_low, _ = find_peaks(-v, height=-(v_min + abs(v_min)*0.1)) # Min付近 (反転してheight)

    # サイクル数推定
    n_cycles = min(len(peaks_high), len(peaks_low))
    
    if n_cycles == 0:
        # 分割失敗時は全データをCycle1とする
        return [{"v": v, "i": i}]

    # 分割実行
    # Start -> Max1 -> Min1 -> Start(Next) という構造を想定
    
    # 最初の開始点
    cycle_start_idx = 0
    
    for k in range(n_cycles):
        # このサイクルのMaxとMinのインデックス
        # 時系列順になっているはず
        p_h = peaks_high[k]
        p_l = peaks_low[k]
        
        # 順番が Max -> Min か Min -> Max かは初期スイープ方向による
        # 終了点を探す: 最後の極値の後、再びInitに戻る点
        last_extremum_idx = max(p_h, p_l)
        
        # last_extremum_idx 以降で、v_init に最も近づく点を次の開始点とする
        search_start = last_extremum_idx + 10
        if search_start >= len(v):
            cycle_end_idx = len(v)
        else:
            # Initとの差分
            diff = np.abs(v[search_start:] - v_init)
            # 最小点を探す (次のサイクルの始まり)
            # ただし、単調減少して近づく場合などを見極める必要がある
            # ここではシンプルに極小値を探す
            local_min_idx = np.argmin(diff)
            cycle_end_idx = search_start + local_min_idx + 1 # +1で含める
        
        # 範囲外ガード
        if cycle_end_idx > len(v): cycle_end_idx = len(v)
        
        # スライス
        v_seg = v[cycle_start_idx:cycle_end_idx]
        i_seg = i[cycle_start_idx:cycle_end_idx]
        cycles.append({"v": v_seg, "i": i_seg})
        
        cycle_start_idx = cycle_end_idx # 次のスタート
        if cycle_start_idx >= len(v) - 10: break

    # 残りカスがあれば統合するか捨てるか...ここでは捨てるか、完全なサイクルのみ返す
    if len(cycles) == 0:
         return [{"v": v, "i": i}]
         
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
    "2️⃣ 個別解析", 
    "3️⃣ 比較・重ね書き", 
    "📝 HOMO/LUMO", 
    "ℹ️ メモ・原理"
])

# ==========================================
# Tab 1: 校正 (変更なし)
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
                # 簡易的な最大・最小
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
                fig.add_vline(x=E_half, line_dash="dash", line_color="green", annotation_text="E 1/2")
                fig = update_fig_layout(fig, f"Standard ({fc_file.name})", "V", "A", show_grid, show_mirror, show_ticks, axis_width, font_size)
                st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Tab 2: 個別解析 (サイクル分割 & 複数ピーク)
# ==========================================
with tab2:
    st.header("サンプルデータの個別解析")
    
    shift_val = st.session_state['calibration_shift']
    if st.session_state['is_calibrated']:
        st.success(f"✅ 現在の補正値: **{shift_val:.4f} V**")
    else:
        st.warning("⚠️ 未校正 (元の電圧表示)")

    if sample_files:
        # ファイル選択
        selected_file_obj = st.selectbox("解析するファイルを選択", sample_files, format_func=lambda x: x.name)
        
        if selected_file_obj:
            # データ読み込み
            df_s = load_data(selected_file_obj, skip_rows, sep=data_sep)
            
            if df_s is not None and df_s.shape[1] >= max(x_col_idx, y_col_idx):
                v_full = df_s.iloc[:, x_col_idx - 1].values
                i_full = df_s.iloc[:, y_col_idx - 1].values
                if smoothing: i_full = smooth_data(i_full)
                v_full_calib = v_full - shift_val

                # --- サイクル分割設定エリア ---
                st.markdown("### 🔄 サイクル分割・表示設定")
                use_cycles = st.checkbox("複数回スキャン（サイクル）として分割する", value=False)
                
                # デフォルトの電圧範囲
                def_init, def_max, def_min = float(v_full[0]), float(np.max(v_full)), float(np.min(v_full))
                
                active_v_calib = v_full_calib
                active_i = i_full
                cycle_info_str = "全データ"

                if use_cycles:
                    col_cy1, col_cy2, col_cy3 = st.columns(3)
                    with col_cy1: c_init = st.number_input("測定 初期電圧 (V)", value=def_init, step=0.1, format="%.2f")
                    with col_cy2: c_max = st.number_input("測定 最大電圧 (V)", value=def_max, step=0.1, format="%.2f")
                    with col_cy3: c_min = st.number_input("測定 最小電圧 (V)", value=def_min, step=0.1, format="%.2f")
                    
                    # 分割実行
                    cycles_data = split_cycles_by_voltage(v_full, i_full, c_init, c_max, c_min)
                    
                    if len(cycles_data) > 0:
                        # サイクル選択
                        cy_options = [f"Cycle {k+1}" for k in range(len(cycles_data))]
                        cy_options.insert(0, "All Cycles (Raw)")
                        selected_cy_label = st.selectbox("表示するサイクルを選択", cy_options)
                        
                        if selected_cy_label != "All Cycles (Raw)":
                            # "Cycle X" を選択
                            cy_idx = int(selected_cy_label.split(" ")[1]) - 1
                            active_v_calib = cycles_data[cy_idx]["v"] - shift_val
                            active_i = cycles_data[cy_idx]["i"]
                            cycle_info_str = f"{selected_cy_label}"
                        else:
                            # 全データ
                            cycle_info_str = "全データ (重ね書き)"
                    else:
                        st.warning("サイクルの分割に失敗しました。条件を見直してください。")

                # --- メイングラフとピーク解析 ---
                st.divider()
                st.subheader(f"📈 解析: {selected_file_obj.name} - [{cycle_info_str}]")
                
                # 左右分割
                col_main_L, col_main_R = st.columns([1, 1])

                with col_main_L:
                    st.markdown("**1. ピーク検索条件**")
                    p_min_def, p_max_def = float(np.min(active_v_calib)), float(np.max(active_v_calib))
                    col_p1, col_p2 = st.columns(2)
                    with col_p1: p_min = st.number_input("探索範囲 Min (V)", value=p_min_def, step=0.1, format="%.2f")
                    with col_p2: p_max = st.number_input("探索範囲 Max (V)", value=p_max_def, step=0.1, format="%.2f")
                    
                    # ピーク検出感度
                    prominence_val = st.slider("ピーク検出感度 (Prominence)", 0.0, 0.5, 0.01, 0.005, help="値を大きくすると、小さなノイズを無視します。")
                    
                    # 範囲内データ抽出
                    mask_range = (active_v_calib >= p_min) & (active_v_calib <= p_max)
                    
                    # 検出実行
                    v_roi = active_v_calib[mask_range]
                    i_roi = active_i[mask_range]
                    
                    detected_peaks_top = []
                    detected_peaks_btm = []

                    if len(v_roi) > 0:
                        detected_peaks_top, detected_peaks_btm = detect_multiple_peaks(v_roi, i_roi, prominence_val)

                    # --- 検出結果の表示と選択 ---
                    st.markdown("**2. 検出されたピーク一覧**")
                    st.caption("登録したいピークにチェックを入れてください")
                    
                    selected_peaks_to_add = []
                    
                    # 酸化ピーク(Top)
                    if detected_peaks_top:
                        st.markdown(f"🔴 **酸化 (極大) ピーク: {len(detected_peaks_top)}個**")
                        for pk in detected_peaks_top:
                            chk = st.checkbox(f"{pk['E']:.3f} V (I={pk['I']:.2e})", value=True, key=f"top_{pk['E']}")
                            if chk: selected_peaks_to_add.append(pk)
                    else:
                        st.info("酸化ピークは見つかりませんでした")

                    # 還元ピーク(Bottom)
                    if detected_peaks_btm:
                        st.markdown(f"🔵 **還元 (極小) ピーク: {len(detected_peaks_btm)}個**")
                        for pk in detected_peaks_btm:
                            chk = st.checkbox(f"{pk['E']:.3f} V (I={pk['I']:.2e})", value=True, key=f"btm_{pk['E']}")
                            if chk: selected_peaks_to_add.append(pk)
                    else:
                        st.info("還元ピークは見つかりませんでした")
                    
                    # 登録ボタン
                    if st.button("選択したピークをリストに保存 💾"):
                        # 単独登録かペア登録か？
                        # ここではシンプルに「検出されたピーク情報」として保存する
                        # ただしE1/2を計算するにはペアが必要。
                        # 今回の要望は「探せるように」なので、個別に保存しつつ、E1/2計算はユーザーに任せるか、
                        # あるいはTop/Bottomの平均を自動で出すか。
                        # ここでは「個別のピーク座標」を保存する形にする。
                        
                        count = 0
                        for pk in selected_peaks_to_add:
                            st.session_state['peak_results'].append({
                                "File": selected_file_obj.name,
                                "Cycle": cycle_info_str,
                                "Type": pk["Type"],
                                "Potential (V)": pk["E"],
                                "Current (A)": pk["I"]
                            })
                            count += 1
                        st.success(f"{count}個のピークを保存しました！")

                with col_main_R:
                    # グラフ描画
                    fig_check = go.Figure()
                    
                    # 全データ (薄く)
                    if use_cycles and cycle_info_str != "全データ (重ね書き)":
                         fig_check.add_trace(go.Scatter(x=v_full_calib, y=i_full, mode='lines', line=dict(color='lightgray'), name="All Data"))
                    
                    # アクティブデータ
                    fig_check.add_trace(go.Scatter(x=active_v_calib, y=active_i, mode='lines', line=dict(color='black', width=2), name="Active Data"))
                    
                    # 探索範囲
                    fig_check.add_trace(go.Scatter(x=v_roi, y=i_roi, mode='lines', line=dict(color='orange', width=4), opacity=0.4, name="Search Range"))
                    
                    # 検出されたピークのプロット (未保存のものも表示)
                    if detected_peaks_top:
                        x_p = [p['E'] for p in detected_peaks_top]
                        y_p = [p['I'] for p in detected_peaks_top]
                        fig_check.add_trace(go.Scatter(x=x_p, y=y_p, mode='markers', marker=dict(color='red', size=10, symbol='circle-open'), name="Detected (Ox)"))
                    
                    if detected_peaks_btm:
                        x_p = [p['E'] for p in detected_peaks_btm]
                        y_p = [p['I'] for p in detected_peaks_btm]
                        fig_check.add_trace(go.Scatter(x=x_p, y=y_p, mode='markers', marker=dict(color='blue', size=10, symbol='circle-open'), name="Detected (Red)"))

                    # 保存済みピークのプロット
                    saved = [p for p in st.session_state['peak_results'] if p['File'] == selected_file_obj.name]
                    if saved:
                        x_s = [p['Potential (V)'] for p in saved]
                        y_s = [p['Current (A)'] for p in saved]
                        fig_check.add_trace(go.Scatter(x=x_s, y=y_s, mode='markers', marker=dict(color='green', size=12, symbol='star'), name="Saved"))

                    fig_check = update_fig_layout(fig_check, f"Analysis: {selected_file_obj.name}", "V vs Fc/Fc+", "Current / A", show_grid, show_mirror, show_ticks, axis_width, font_size)
                    st.plotly_chart(fig_check, use_container_width=True)

                # --- 保存リストの表示 ---
                st.divider()
                st.markdown("### 📋 保存されたピークリスト")
                if st.session_state['peak_results']:
                    res_df = pd.DataFrame(st.session_state['peak_results'])
                    st.dataframe(res_df, use_container_width=True)
                    
                    # E1/2 計算ツール (簡易版)
                    st.markdown("**🛠️ E1/2 簡易計算機**")
                    col_calc1, col_calc2, col_calc3 = st.columns(3)
                    
                    # ファイル内の酸化・還元ピークを抽出して選択肢にする
                    current_file_peaks = res_df[res_df['File'] == selected_file_obj.name]
                    ox_opts = current_file_peaks[current_file_peaks['Type'].str.contains("Anodic")]['Potential (V)'].tolist()
                    red_opts = current_file_peaks[current_file_peaks['Type'].str.contains("Cathodic")]['Potential (V)'].tolist()
                    
                    sel_ox = col_calc1.selectbox("酸化ピークを選択", ox_opts) if ox_opts else None
                    sel_red = col_calc2.selectbox("還元ピークを選択", red_opts) if red_opts else None
                    
                    if sel_ox is not None and sel_red is not None:
                        calc_half = (sel_ox + sel_red) / 2
                        col_calc3.metric("計算された E1/2", f"{calc_half:.3f} V")
                    else:
                        col_calc3.info("酸化・還元ピークをリストから選択してください")

                    if st.button("リストをクリア 🗑️"):
                        st.session_state['peak_results'] = []
                        st.rerun()

    else:
        st.info("👈 サイドバーからサンプルデータをアップロードしてください。")

# ==========================================
# Tab 3: 比較・重ね書き (変更なし)
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
# Tab 4 & 5 (省略なし)
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

with tab5:
    st.header("📝 メモ・原理")
    st.markdown(EXPLANATION_TEXT)