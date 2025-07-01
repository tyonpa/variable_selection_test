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
st.title('Variable Selection :blue[Test] :sunglasses:')


#  options
## input data sectioc
input_data_error_flag = False
df_files_name = ['test_california_data']
df_files = [df_california]

st.subheader('Select input data')

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

st.subheader('Select using model')

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

st.subheader('Select valiable selection method')

selected_filter_method = st.multiselect('filter method', vs_methods_filter, default=vs_methods_filter[0])
selected_wrapper_method = st.multiselect('wrapper method', vs_methods_wrapper, default=vs_methods_wrapper[0])
selected_embedded_method = st.multiselect('embedded method', vs_methods_embedded, default=vs_methods_embedded[1])

if (selected_filter_method == []) and (selected_wrapper_method == []) and (selected_embedded_method == []):
    st.warning('少なくとも１つ変数選択手法を選択してください。')
    vs_error_flag = True

expander_data = st.expander('データの前処理の詳細設定')
expander_method = st.expander("変数選択手法の詳細設定")

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
            
    
expander_method.text('しょうさいせっていだお')

st.divider()


## start button
analysis_start_flag = False
result = []

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

    for model in selected_model:
        _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test)
        result.append(['Nothing', f'{model}', test_score, rmse, num_ferature])
        
        for f_method in selected_filter_method:
            if f_method == 'VIF':
                _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, filter_vif(x_train, vif))
                result.append(['VIF', f'{model}', test_score, rmse, num_ferature])
            if f_method == 'MI':
                _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, filter_MI(x_train, y_train, percentile_MI))
                result.append(['MI', f'{model}', test_score, rmse, num_ferature])
            if f_method == 'ANOVA':
                _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, filter_ANOVA(x_train, y_train, percentile_ANOVA))
                result.append(['ANOVA', f'{model}', test_score, rmse, num_ferature])
        
        for w_method in selected_wrapper_method:
            with st.spinner(f'{model} RFECV...', show_time=True):
                if w_method == 'RFECV':
                    if model == 'SVR':
                        _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV(call_model['LinearRegression'], x_train, y_train, cv_RFE))
                    else:
                        _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV(call_model[model], x_train, y_train, cv_RFE))
                    result.append(['RFECV', f'{model}', test_score, rmse, num_ferature])
            with st.spinner(f'{model} RFECV_sensitivity...', show_time=True):
                if w_method == 'RFECV_sensitivity':
                    _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV_sensitivity(call_model[model], x_train, y_train, cv_RFE_sensitivity, score_width_sensitivity))
                    result.append(['RFECV_sensitivity', f'{model}', test_score, rmse, num_ferature])
            with st.spinner(f'{model} RFECV_MI...', show_time=True):
                if w_method == 'RFECV_MI':
                    _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV_MI(call_model[model], x_train, y_train, cv_RFE_MI, score_width_MI))
                    result.append(['RFECV_MI', f'{model}', test_score, rmse, num_ferature])
            with st.spinner(f'{model} RFECV_SHAP...', show_time=True):
                if w_method == 'RFECV_SHAP':
                    _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, wrapper_RFECV_SHAP(call_model[model], x_train, y_train, cv_RFE_SHAP, score_width_SHAP))
                    result.append(['RFECV_SHAP', f'{model}', test_score, rmse, num_ferature])

        for e_method in selected_embedded_method:
            if e_method == 'LassoCV':
                _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, embbeded_lasso(x_train, y_train))
                result.append(['LassoCV', f'{model}', test_score, rmse, num_ferature])
            if e_method == 'ElasticNetCV':
                _, test_score, rmse, num_ferature = feature_selection_by_method(call_model[model], x_train, y_train, x_test, y_test, embbeded_elasticnet(x_train, y_train))
                result.append(['ElasticNetCV', f'{model}', test_score, rmse, num_ferature])

    st.dataframe(pd.DataFrame(result, columns=['Method', 'Model', 'R2', 'RMSE', 'Num_feature']))