import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from method import model_evaluation, feature_selection_by_method, custom_rfecv
from method import filter_vif, filter_MI, filter_ANOVA
from method import wrapper_RFECV, wrapper_RFECV_sensitivity, wrapper_RFECV_MI, wrapper_RFECV_SHAP
from method import embbeded_lasso, embbeded_elasticnet


#  data
california_housing = fetch_california_housing()
train_x = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
train_y = pd.Series(california_housing.target)
df_california = pd.concat([train_x, train_y], axis=1)
df_california.columns = df_california.columns.tolist()[:-1] + ["target"]

model_lr = LinearRegression()
model_rf = RandomForestRegressor()
model_SVR = SVR()
model_lgb = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=100,
    learning_rate=0.1,
    random_state=0, verbose=-1)

call_model = {'LinearRegression': model_lr, 'RandomForestRegressor': model_rf, 'SVR': model_SVR, 'lightGBM': model_lgb}

#  app
st.markdown('<h1 style="text-align: center; ">特徴量選択<span style="color: skyblue;">テスター</span>😎<h1>', unsafe_allow_html=True)
st.set_page_config(page_title="特徴量選択テスター")

#  options
## input data sectioc
input_data_error_flag = False
df_files_name = ['test_california_data']
df_files = [df_california]

st.subheader('データインプット', help='Select input data')

input_data_col1, input_data_col2 = st.columns([1, 2])
with input_data_col1:
    uploaded_csv = st.file_uploader('upload csv', type='csv')
    if uploaded_csv != None:
        try:
            # ファイルの内容を読み込む
            df_csv = pd.read_csv(uploaded_csv)
            
            # 空のデータフレームかどうかをチェック
            if df_csv.empty:
                st.warning("アップロードされたCSVファイルにはデータが含まれていません。")
                input_data_error_flag = True
            elif df_csv.shape[1] < 3:
                st.warning("データの列数が少なすぎます。")
                input_data_error_flag = True
            else:
                st.success("CSVファイルが正常に読み込まれました！")
                df_files_name.append(uploaded_csv.name[:-4])
                df_files.append(df_csv)

        except pd.errors.EmptyDataError:
            st.error("アップロードされたファイルには読み取れるデータがありません（EmptyDataError）。")
            input_data_error_flag = True
        except pd.errors.ParserError:
            st.error("CSVの解析中にエラーが発生しました（ParserError）。")
            input_data_error_flag = True
        except Exception as e:
            st.error(f"予期しないエラーが発生しました: {e}")
            input_data_error_flag = True

with input_data_col2:
    selected_csv = st.selectbox(
        'select_data', 
        df_files_name
    )
    if df_files[df_files_name.index(selected_csv)].shape[0] > 10_000:
        st.dataframe(df_files[df_files_name.index(selected_csv)].head(10_000), height=250)
    else:
        st.dataframe(df_files[df_files_name.index(selected_csv)], height=250)

st.divider()


## select model section
models_error_flag = False
models = ['LinearRegression', 'RandomForestRegressor', 'SVR', 'lightGBM']

st.subheader('機械学習モデルの選択', help='Select using machine learning model')

selected_model = st.multiselect('select model', models, default=models)

if selected_model == []:
    st.warning('少なくとも１つモデルを選択してください。')
    models_error_flag = True

st.divider()


## select variable selection method section
vs_error_flag = False
vs_methods_filter = ['VIF', 'MI', 'ANOVA']
vs_methods_wrapper = ['RFECV', 'RFECV_sensitivity', 'RFECV_MI', 'RFECV_SHAP']
vs_methods_embedded = ['LassoCV', 'ElasticNetCV']

st.subheader('特徴量選択手法の選択', help='Select valiable selection method')

selected_filter_method = st.multiselect('filter method', vs_methods_filter, default=vs_methods_filter[0])
selected_wrapper_method = st.multiselect('wrapper method', vs_methods_wrapper, default=vs_methods_wrapper[0])
selected_embedded_method = st.multiselect('embedded method', vs_methods_embedded, default=vs_methods_embedded[1])

if (selected_filter_method == []) and (selected_wrapper_method == []) and (selected_embedded_method == []):
    st.warning('少なくとも１つ特徴量選択手法を選択してください。')
    vs_error_flag = True

expander_data = st.expander('データの前処理の詳細設定')
expander_method = st.expander("特徴量選択手法の詳細設定")

if input_data_error_flag != True:
    if df_files_name.index(selected_csv) != 0:
        num_y = expander_data.slider("y（目的変数）とする列番号", 1, df_files[df_files_name.index(selected_csv)].shape[1]) - 1
    num_test_size = expander_data.slider("ホールドアウト テストデータサイズ", 0.0, 1.0, value=0.3)
    is_scale = expander_data.toggle("標準化", value=True)
    if expander_data.button("前処理テスト"):
        try:
            if df_files_name.index(selected_csv) == 0:
                df_X = train_x
                df_y = train_y
            else:
                df_X = df_files[df_files_name.index(selected_csv)].drop(columns=[df_files[df_files_name.index(selected_csv)].columns[num_y]])
                df_y = df_files[df_files_name.index(selected_csv)].iloc[:, num_y]

            x_train, x_test, y_train, y_test = train_test_split(df_X, df_y, test_size=num_test_size, random_state=0)
            
            if is_scale:
                scaler = StandardScaler()
                x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
                x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
            
            expander_data.info(f"元ファイル：{df_files[df_files_name.index(selected_csv)].shape}, 学習データX：{x_train.shape}, 学習データy：{y_train.shape}")
            expander_data.success("正常に終了しました。")
            expander_data.dataframe(pd.DataFrame(x_train).head(3))

        except Exception as e:
            st.error(f"予期しないエラーが発生しました: {e}")

if vs_error_flag != True and models_error_flag != True:
    if 'VIF' in selected_filter_method:
        expander_method.subheader('filter method VIF')
        vif = expander_method.slider("VIF", 1, 20, value=5, key='VIF')
    if 'MI' in selected_filter_method:
        expander_method.subheader('filter method MI')
        percentile_MI = expander_method.slider('選択する特徴量の割合', 1, 100, value=30, key='MI')
    if 'ANOVA' in selected_filter_method:
        expander_method.subheader('filter method ANOVA')
        percentile_ANOVA = expander_method.slider('選択する特徴量の割合', 1, 100, value=30, key='ANOVA')
    if 'RFECV' in selected_wrapper_method:
        expander_method.subheader('wrapper method RFECV')
        cv_RFE = expander_method.slider('CV', 2, 10, value=5, key='RFECV')
    if 'RFECV_sensitivity' in selected_wrapper_method:
        expander_method.subheader('wrapper method RFECV_sensitivity')
        cv_RFE_sensitivity = expander_method.slider('CV', 2, 10, value=5, key='RFECV_sensitivity_cv')
        score_width_sensitivity = expander_method.slider('誤差許容幅', 0.0, 1.0, value=0.0, key='RFECV_sensitivity_width')
    if 'RFECV_MI' in selected_wrapper_method:
        expander_method.subheader('wrapper method RFECV_MI')
        cv_RFE_MI = expander_method.slider('CV', 2, 10, value=5, key='RFECV_MI_cv')
        score_width_MI = expander_method.slider('誤差許容幅', 0.0, 1.0, value=0.0, key='RFECV_MI_width')
    if 'RFECV_SHAP' in selected_wrapper_method:
        expander_method.subheader('wrapper method RFECV_SHAP')
        cv_RFE_SHAP = expander_method.slider('CV', 2, 10, value=5, key='RFECV_SHAP_cv')
        score_width_SHAP = expander_method.slider('誤差許容幅', 0.0, 1.0, value=0.0, key='RFECV_SHAP_width')
            
st.divider()


## start button
analysis_start_flag = False
all_tasks = (len(selected_filter_method) + len(selected_wrapper_method) + len(selected_embedded_method) + 1) * len(selected_model)
completed_tasks = 0
result_csv = []
result_col = []

### 画面再読み込み時に更新されないようにするデータ
if 'df_X' not in st.session_state:
    st.session_state.df_X = None
if 'result_csv' not in st.session_state:
    st.session_state.result_csv = None
if 'result_col' not in st.session_state:
    st.session_state.result_col = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = None
if 'selected_methods' not in st.session_state:
    st.session_state.selected_methods = None
if 'finish_analysis' not in st.session_state:
    st.session_state.finish_analysis = False

_, middle, _ = st.columns(3)
if middle.button("start", use_container_width=True):
    if models_error_flag or vs_error_flag == True:
        middle.warning('エラーを解決してから実行してください。')
    else:
        analysis_start_flag = True

if analysis_start_flag:
    try:
        if df_files_name.index(selected_csv) == 0:
            df_X = train_x
            df_y = train_y
        else:
            df_X = df_files[df_files_name.index(selected_csv)].drop(columns=[df_files[df_files_name.index(selected_csv)].columns[num_y]])
            df_y = df_files[df_files_name.index(selected_csv)].iloc[:, num_y]

        x_train, x_test, y_train, y_test = train_test_split(df_X, df_y, test_size=num_test_size, random_state=0)
        
        if is_scale:
            scaler = StandardScaler()
            x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
            x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    except Exception as e:
        st.error(f"前処理で予期しないエラーが発生しました: {e}")

    my_bar = st.progress(0, '実行中...')
    
    for model in selected_model:
        feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test)
        result_csv.append(['Normal(no selection)', f'{model}', test_score, rmse, num_ferature])
        result_col.append(feat_col)
        completed_tasks += 1
        my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
        
        for f_method in selected_filter_method:
            if f_method == 'VIF':
                feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, filter_vif(x_train, vif))
                result_csv.append(['VIF', f'{model}', test_score, rmse, num_ferature])
                result_col.append(feat_col)
                completed_tasks += 1
                my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
            if f_method == 'MI':
                feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, filter_MI(x_train, y_train, percentile_MI))
                result_csv.append(['MI', f'{model}', test_score, rmse, num_ferature])
                result_col.append(feat_col)
                completed_tasks += 1
                my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
            if f_method == 'ANOVA':
                feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, filter_ANOVA(x_train, y_train, percentile_ANOVA))
                result_csv.append(['ANOVA', f'{model}', test_score, rmse, num_ferature])
                result_col.append(feat_col)
                completed_tasks += 1
                my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
        
        for w_method in selected_wrapper_method:
            with st.spinner(f'{model} RFECV...', show_time=True):
                if w_method == 'RFECV':
                    if model == 'SVR':
                        feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV(call_model['LinearRegression'], x_train, y_train, cv_RFE))
                    else:
                        feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV(call_model[model], x_train, y_train, cv_RFE))
                    result_csv.append(['RFECV', f'{model}', test_score, rmse, num_ferature])
                    result_col.append(feat_col)
                    completed_tasks += 1
                    my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
            with st.spinner(f'{model} RFECV_sensitivity...', show_time=True):
                if w_method == 'RFECV_sensitivity':
                    feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV_sensitivity(call_model[model], x_train, y_train, cv_RFE_sensitivity, score_width_sensitivity))
                    result_csv.append(['RFECV_sensitivity', f'{model}', test_score, rmse, num_ferature])
                    result_col.append(feat_col)
                    completed_tasks += 1
                    my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
            with st.spinner(f'{model} RFECV_MI...', show_time=True):
                if w_method == 'RFECV_MI':
                    feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV_MI(call_model[model], x_train, y_train, cv_RFE_MI, score_width_MI))
                    result_csv.append(['RFECV_MI', f'{model}', test_score, rmse, num_ferature])
                    result_col.append(feat_col)
                    completed_tasks += 1
                    my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
            with st.spinner(f'{model} RFECV_SHAP...', show_time=True):
                if w_method == 'RFECV_SHAP':
                    feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV_SHAP(call_model[model], x_train, y_train, cv_RFE_SHAP, score_width_SHAP))
                    result_csv.append(['RFECV_SHAP', f'{model}', test_score, rmse, num_ferature])
                    result_col.append(feat_col)
                    completed_tasks += 1
                    my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')

        for e_method in selected_embedded_method:
            with st.spinner(f'{model} LassoCV...', show_time=True):
                if e_method == 'LassoCV':
                    feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, embbeded_lasso(x_train, y_train))
                    result_csv.append(['LassoCV', f'{model}', test_score, rmse, num_ferature])
                    result_col.append(feat_col)
                    completed_tasks += 1
                    my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
            with st.spinner(f'{model} ElasticNetCV...', show_time=True):
                if e_method == 'ElasticNetCV':
                    feat_col, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, embbeded_elasticnet(x_train, y_train))
                    result_csv.append(['ElasticNetCV', f'{model}', test_score, rmse, num_ferature])
                    result_col.append(feat_col)
                    completed_tasks += 1
                    my_bar.progress(int(completed_tasks/all_tasks * 100), '実行中...')
    
    st.session_state.df_X = df_X
    st.session_state.result_csv = result_csv
    st.session_state.result_col = result_col
    st.session_state.selected_models = selected_model
    st.session_state.selected_methods = ['Normal'] + selected_filter_method + selected_wrapper_method + selected_embedded_method
    st.session_state.finish_analysis = True

## analysis results
if st.session_state.finish_analysis:
    selected_models = st.session_state.selected_models
    selected_methods = st.session_state.selected_methods
    meta_df = st.session_state.df_X
    
    # analysis result
    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader("分析結果")
    st.dataframe(pd.DataFrame(st.session_state.result_csv, columns=['Method', 'Model', 'R2', 'RMSE', 'Num_feature']), hide_index=True)
    
    # show selected features
    expander_result = st.expander("選択された特徴量")
    is_show_rej_col = expander_result.toggle("除外した特徴量を表示する", value=True)
    for i in range(len(st.session_state.selected_models)):
        expander_result.subheader(f"{st.session_state.selected_models[i]}")
        for j in range(len(st.session_state.selected_methods)):
            picked_model = st.session_state.selected_models[i]
            picked_method = st.session_state.selected_methods[j]
            expander_result.text(picked_method)
            sel_col = st.session_state.result_col[selected_methods.index(picked_method) + len(selected_methods) * selected_models.index(picked_model)].tolist()
            all_col = meta_df.columns.tolist()
            markdown_col = ""
            for col in all_col:
                if col in sel_col:
                    markdown_col += f":blue-badge[{col}] "
                elif col not in sel_col and is_show_rej_col:
                    markdown_col += f":gray-badge[{col}] "
            expander_result.markdown(markdown_col)
    
    # output analysis result
    df_res = pd.DataFrame(st.session_state.result_csv)
    df_csv = pd.DataFrame(st.session_state.result_col)
    df_con = pd.concat([df_res, df_csv], axis=1)
    df_con.columns = ['Method', 'Model', 'R2', 'RMSE', 'Num_feature'] + [f'col_{i}' for i in range(len(all_col))]
    
    st.download_button(
                    label="分析結果の出力（csv）",
                    data=df_con.to_csv().encode("utf-8"),
                    file_name=f"VariableSelection_Conclusion_{df_files_name[-1]}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                    )

    # output dataset
    st.markdown('<br>', unsafe_allow_html=True)
    st.subheader("特徴量選択済みデータの出力")
    output_data_col1, output_data_col2 = st.columns(2)
    with output_data_col1:
        output_model = st.selectbox('select model', selected_models)
    
    with output_data_col2:
        output_method = st.selectbox('select method', selected_methods)
    
    with st.spinner(f'CSV 準備中...', show_time=True):
        output_col = st.session_state.result_col[selected_methods.index(output_method) + len(selected_methods) * selected_models.index(output_model)]
        output_csv = meta_df.loc[:, output_col.tolist()].to_csv().encode("utf-8")

    st.download_button(
                    label="特徴量選択済みデータの出力（csv）",
                    data=output_csv,
                    file_name=f"{output_model}-{output_method}_{df_files_name[-1]}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                    )
