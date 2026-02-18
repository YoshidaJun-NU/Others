import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def detect_header_row(file_path_or_buffer, encoding):
    """ファイル内のデータ開始位置を自動検出"""
    header_row = 0
    keywords = ['[Data]', 'XYDATA', 'Wavelength'] # 対応キーワード
    try:
        if isinstance(file_path_or_buffer, str):
            with open(file_path_or_buffer, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
        else:
            file_path_or_buffer.seek(0)
            content = file_path_or_buffer.read().decode(encoding, errors='ignore')
            lines = content.splitlines()
            file_path_or_buffer.seek(0)

        for i, line in enumerate(lines):
            if any(key in line for key in keywords):
                header_row = i + 1
                break
    except Exception:
        pass 
    return header_row

def load_data_robust(file_path_or_buffer, sep, header, encoding):
    encodings_to_try = [encoding, 'utf-8', 'cp932', 'shift_jis', 'utf-8-sig']
    last_error = None
    for enc in encodings_to_try:
        try:
            if isinstance(file_path_or_buffer, str):
                df = pd.read_csv(file_path_or_buffer, sep=sep, header=header, encoding=enc, engine='python')
            else:
                file_path_or_buffer.seek(0)
                df = pd.read_csv(file_path_or_buffer, sep=sep, header=header, encoding=enc, engine='python')
            return df
        except Exception as e:
            last_error = e
            continue
    raise last_error

# ---------------------------------------------------------
# アプリ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Advanced DSC Plotter", layout="wide")
st.title("DSC Graph Plotter")

# --- サイドバー: 設定 ---
st.sidebar.header("1. データ読み込み")
uploaded_file = st.sidebar.file_uploader("ファイルをアップロード", type=['csv', 'txt'])

demo_file_path = "demoDSC.txt"
target_file = uploaded_file if uploaded_file else (demo_file_path if os.path.exists(demo_file_path) else None)

if target_file:
    # --- 読み込み詳細 ---
    with st.sidebar.expander("インポート設定", expanded=False):
        encoding_option = st.selectbox("文字コード", ["utf-8", "cp932", "shift_jis"])
        delimiter = st.radio("区切り文字", [", (CSV)", "\\t (Tab)", "Space"], index=1)
        sep = "," if delimiter == ", (CSV)" else "\t" if delimiter == "\\t (Tab)" else r"\s+"
        def_head = detect_header_row(target_file, encoding_option)
        header_arg = st.number_input("ヘッダー行番号", min_value=0, value=def_head)

    try:
        df_raw = load_data_robust(target_file, sep, header_arg, encoding_option)
        df = df_raw.apply(pd.to_numeric, errors='coerce').dropna(how='all').dropna().reset_index(drop=True)
        
        columns = df.columns.tolist()
        st.sidebar.subheader("2. グラフ設定")
        col_x = st.sidebar.selectbox("X軸列", columns, index=1 if len(columns)>1 else 0)
        col_y = st.sidebar.selectbox("Y軸列", columns, index=2 if len(columns)>2 else 0)

        # --- スタイル設定 ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. 表示スタイル")
        tick_dir = st.sidebar.radio("目盛の向き", ["in (内向き)", "out (外向き)"], index=1, horizontal=True).split()[0]
        line_width = st.sidebar.slider("線の太さ", 0.5, 5.0, 1.5, 0.5)
        font_size = st.sidebar.slider("文字の大きさ", 8, 24, 12, 1)
        
        # --- ラベル・範囲 ---
        x_lab = st.sidebar.text_input("X軸ラベル", "Temperature (℃)")
        y_lab = st.sidebar.text_input("Y軸ラベル", "DSC (mW)")
        
        c_x1, c_x2 = st.sidebar.columns(2)
        x_min = c_x1.number_input("X最小", value=float(df[col_x].min()))
        x_max = c_x2.number_input("X最大", value=float(df[col_x].max()))

        # --- メインレイアウト ---
        # グラフプレビュー
        graph_area = st.container()
        st.divider()
        settings_area = st.container()

        # プロット個別設定
        plot_configs = []
        with settings_area:
            st.subheader("プロット範囲・オフセット設定")
            n_plots = st.number_input("プロット数", 1, 10, 2)
            s_cols = st.columns(2)
            
            for i in range(n_plots):
                with s_cols[i % 2]:
                    with st.expander(f"Curve {i+1} の設定", expanded=True):
                        total = len(df)
                        # デフォルト値の自動割り振り
                        s_def = [30, 800][i] if i < 2 else 0
                        e_def = [700, 1750][i] if i < 2 else total
                        
                        # --- 変更点1: 表示切り替えチェックボックスの追加 ---
                        is_visible = st.checkbox(f"Curve {i+1} を表示", value=True, key=f"visible_{i}")
                        
                        c1, c2 = st.columns(2)
                        start = c1.number_input(f"開始行", 0, total, s_def, key=f"s{i}")
                        end = c2.number_input(f"終了行", 0, total, e_def, key=f"e{i}")
                        
                        c3, c4 = st.columns(2)
                        color = c3.color_picker(f"色", ["#FF4B4B", "#1F77B4"][i] if i < 2 else "#333333", key=f"c{i}")
                        offset = c4.number_input(f"Yオフセット", value=0.0, step=0.1, key=f"o{i}")
                        
                        # 設定辞書に visible フラグを追加
                        plot_configs.append({
                            "start": start, 
                            "end": end, 
                            "color": color, 
                            "offset": offset, 
                            "label": f"Scan {i+1}",
                            "visible": is_visible
                        })

        # グラフ描画実行
        with graph_area:
            _, center_col, _ = st.columns([0.2, 0.6, 0.2])
            
            with center_col:
                plt.rcParams['font.size'] = font_size
                fig, ax = plt.subplots(figsize=(6, 4))
                
                ax.tick_params(direction=tick_dir, top=True, right=True)
                
                for config in plot_configs:
                    # --- 変更点2: visibleがTrueの場合のみプロット ---
                    if config["visible"]:
                        sub = df.iloc[config["start"]:config["end"]]
                        if not sub.empty:
                            ax.plot(sub[col_x], sub[col_y] + config["offset"], 
                                    color=config["color"], linewidth=line_width, label=config["label"])
                
                ax.set_xlim(x_min, x_max)
                ax.set_xlabel(x_lab)
                ax.set_ylabel(y_lab)
                
                # 凡例（表示されているデータがある場合のみ表示）
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(frameon=False, fontsize=font_size*0.8)
                
                st.pyplot(fig)
                
                # スクリプトダウンロード
                st.download_button("Gnuplot用スクリプトを保存", "...", file_name="dsc_plot.plt")

    except Exception as e:
        st.error(f"データの処理中にエラーが発生しました: {e}")

else:
    st.info("左側のサイドバーからデータをアップロードしてください。")

# ---------------------------------------------------------
# 使い方説明
# ---------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.subheader("📖 使い方")
cols = st.columns(3)
with cols[0]:
    st.markdown("**1. データのインポート**")
    st.caption("JASCO形式やCSV形式に対応しています。ヘッダー行は自動検出されますが、ズレる場合は手動で調整してください。")
with cols[1]:
    st.markdown("**2. スタイルの調整**")
    st.caption("論文用には目盛を 'in' に、プレゼン用には文字サイズを大きく設定するのがおすすめです。")
with cols[2]:
    st.markdown("**3. 複数スキャンの分割・表示切替**")
    st.caption("行番号で分割し、オフセットで見やすく配置できます。「Curve X を表示」チェックボックスで特定の曲線を一時的に隠せます。")

plt.close('all')