# streamlit run app.py
# pip install -r requirements.txt

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# st.title('Fraud Detection')
# st.markdown("<h1 style='text-align: center; color: black;'>Churn Prediction</h1>", unsafe_allow_html=True)
im = Image.open("cover.png")
st.image(im, width=700)

html_temp = """
<div style="width:700px;background-color:maroon;padding:10px">
<h1 style="color:white;text-align:center;">Machine Learning Application (Demo)</h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown("<h3></h3>", unsafe_allow_html=True)

def main():
    st.sidebar.header("How would you like to predict?")
    add_selectbox = st.sidebar.selectbox("", ("Unique Input", "Batch Input"))

    features = ['lor', 'mou_Mean', 'change_mou', 'infobase', 'change_rev',
       'eqpdays', 'totmrc_Mean', 'months', 'avgmou', 'mou_peav_Mean',
       'prizm_social_one', 'totrev', 'numbcars', 'crclscod',
       'avgrev', 'avgqty', 'rev_Mean', 'unan_vce_Mean', 'totcalls',
       'drop_blk_Mean','opk_vce_Mean', 'income', 'ovrmou_Mean', 'mou_cvce_Mean',
       'HHstatin', 'uniqsubs', 'mou_opkv_Mean', 'drop_vce_Mean',
       'mouiwylisv_Mean', 'mouowylisv_Mean', 'ethnic', 'dwllsize',
       'ownrent', 'blck_vce_Mean', 'mou_rvce_Mean', 'hnd_price',
       'recv_vce_Mean', 'area', 'adults', 'plcd_vce_Mean',
       'owylis_vce_Mean', 'iwylis_vce_Mean', 'roam_Mean', 'ccrndmou_Mean',
       'custcare_Mean', 'da_Mean', 'marital', 'hnd_webcap',
       'callwait_Mean', 'threeway_Mean', 'phones', 'asl_flag',
       'datovr_Mean', 'actvsubs', 'mou_pead_Mean', 'mou_cdat_Mean',
       'kid16_17', 'creditcd', 'models', 'dualband', 'truck',
       'kid0_2', 'plcd_dat_Mean', 'kid3_5', 'new_cell',
       'kid11_15', 'rv', 'forgntvl', 'drop_dat_Mean', 'refurb_new',
       'kid6_10', 'recv_sms_Mean', 'unan_dat_Mean', 'blck_dat_Mean',
       'callfwdv_Mean']

    cat_cols = ['infobase','prizm_social_one','crclscod']+ \
               ['HHstatin', 'ethnic','dwllsize', 'ownrent', 'area', 'marital', 
                'hnd_webcap', 'asl_flag','kid16_17', 'creditcd','dualband',
                'kid0_2', 'kid3_5', 'new_cell', 'kid11_15','refurb_new', 'kid6_10']
    
    ordered_columns=['rev_Mean', 'mou_Mean', 'totmrc_Mean', 'da_Mean', 'ovrmou_Mean',
       'datovr_Mean', 'roam_Mean', 'change_mou', 'change_rev', 'drop_vce_Mean',
       'drop_dat_Mean', 'blck_vce_Mean', 'blck_dat_Mean', 'unan_vce_Mean',
       'unan_dat_Mean', 'plcd_vce_Mean', 'plcd_dat_Mean', 'recv_vce_Mean',
       'recv_sms_Mean', 'custcare_Mean', 'ccrndmou_Mean', 'threeway_Mean',
       'mou_cvce_Mean', 'mou_cdat_Mean', 'mou_rvce_Mean', 'owylis_vce_Mean',
       'mouowylisv_Mean', 'iwylis_vce_Mean', 'mouiwylisv_Mean',
       'mou_peav_Mean', 'mou_pead_Mean', 'opk_vce_Mean', 'mou_opkv_Mean',
       'drop_blk_Mean', 'callfwdv_Mean', 'callwait_Mean', 'months', 'uniqsubs',
       'actvsubs', 'totcalls', 'totrev', 'avgrev', 'avgmou', 'avgqty',
       'hnd_price', 'phones', 'models', 'truck', 'rv', 'lor', 'adults',
       'income', 'numbcars', 'forgntvl', 'eqpdays', 'new_cell_freq',
       'crclscod_freq', 'asl_flag_freq', 'prizm_social_one_freq', 'area_freq',
       'dualband_freq', 'refurb_new_freq', 'hnd_webcap_freq', 'ownrent_freq',
       'marital_freq', 'infobase_freq', 'HHstatin_freq', 'dwllsize_freq',
       'ethnic_freq', 'kid0_2_freq', 'kid3_5_freq', 'kid6_10_freq',
       'kid11_15_freq', 'kid16_17_freq', 'creditcd_freq']

    # @st.cache
    # bir buyuk bir datatyi read_csv ile tekrar tekrar okutmamak icin hafuzada tutmasi icin st.cache kullanilir.
    lightGBM = pickle.load(open("LightGBM.pkl","rb"))
    
    with open('FE_dict.pkl', 'rb') as handle:
        FE_dict = pickle.load(handle)
        
    with open('limits_dict.pkl', 'rb') as handle:
        limits_dict = pickle.load(handle)
        
    with open('cat_cols_unique_list.pkl', 'rb') as handle:
        cat_cols_unique_list = pickle.load(handle)
        
    with open('description.pkl', 'rb') as handle:
        description = pickle.load(handle)

    if add_selectbox == "Unique Input":
        st.markdown("""
 :dart: Top 10 Most Important Features:\n
""")
        st.sidebar.info(':dart: Low-Importance Features:')
        my_dict = {col:np.nan for col in features}

        # Numeric Features
        for i,col in enumerate(features):
            if (col not in cat_cols) and (i<10):
                my_dict[col] = st.slider(f"{col} ({description[col]}):", int(np.floor(limits_dict[col][0])), int(np.ceil(limits_dict[col][1])), int(limits_dict[col][2]), step=1)
            elif (col not in cat_cols) and (i>=10):
                my_dict[col] = st.sidebar.slider(f"{col} ({description[col]}):", int(np.floor(limits_dict[col][0])), int(np.ceil(limits_dict[col][1])), int(limits_dict[col][2]), step=1)
                
        # Categoric Features
        for i, col in enumerate(cat_cols):
            if i<1:
                my_dict[col] = st.selectbox(f"{col} ({description[col]}):", cat_cols_unique_list[col])
                my_dict[col+"_freq"] = FE_dict[col+"_freq"][my_dict[col]]
            else:
                my_dict[col] = st.sidebar.selectbox(f"{col} ({description[col]}):", cat_cols_unique_list[col])
                my_dict[col+"_freq"] = FE_dict[col+"_freq"][my_dict[col]]
                
                
        df = pd.DataFrame([my_dict]).drop(cat_cols, axis=1)            

        df = df.reindex(columns=ordered_columns, fill_value=0)

        # Table
        def single_customer(my_dict):
            df_table = pd.DataFrame.from_dict([my_dict])
        #     st.table(df_table) 
            st.write('')
            st.dataframe(data=df_table, width=700, height=400)
            st.write('')

        single_customer(my_dict)

        # Button
        if st.button("Submit Manuel Inputs"):
            import time
            with st.spinner("ML Model is loading..."):
                my_bar=st.progress(0)
                for p in range(0,101,10):
                    my_bar.progress(p)
                    time.sleep(0.1)

                    churn_probability = lightGBM.predict_proba(df)
                    is_churn= lightGBM.predict(df)

                st.success(f'The Churn Probability of the Customer is {round(churn_probability[0][1]*100,3)}%')
        #         st.success(f'The Fraud Probability of the Transaction is {fraud_probability[0][1]}')

                if is_churn[0]:
                    st.success("The Customer is CHURN")
                else:
                    st.warning("The Customer is NOT CHURN")
    
    else:
        # Upload a csv
        output = pd.DataFrame()
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            file = pd.read_csv(uploaded_file, index_col=[0])
            flag=file.copy()
            st.dataframe(data=file, width=700, height=1000)
            st.write('')
        #  st.table(file)
        
        # Load Button
        if st.button("Submit CSV File"):
            import time
            with st.spinner("ML Model is loading..."):
                my_bar=st.progress(0)
                for p in range(0,101,10):
                    my_bar.progress(p)
                    time.sleep(0.1)

            for i in file.index:
                for col in cat_cols:
                    file.loc[i,col+"_freq"]  = FE_dict[col+'_freq'][file.loc[i,col]]       

            file = file.drop(cat_cols, axis=1)
            file = file.reindex(columns=ordered_columns, fill_value=0)
            pred_file= pd.DataFrame(lightGBM.predict_proba(file))[[1]].rename({1:'Prediction'}, axis=1)
            pred_file['isChurn'] = pred_file.iloc[:,0].apply(lambda x: 'YES' if np.float(x)>=0.5 else 'NO')
            output = pd.concat([pred_file,flag], axis=1).reset_index().drop('index', axis=1)
            st.write('')
            st.dataframe(data=output, width=700, height=400)
            st.write('')

        def download_link(object_to_download, download_filename, download_link_text):
            if isinstance(object_to_download,pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(object_to_download.encode()).decode()

            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

        # if st.button('Download Output as CSV'):
        tmp_download_link = download_link(output, 'output.csv', 'Click here to download output as csv!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

        comment = st.text_input('Write your comments below.')
        # st.write(comment)

        # if st.button('Download input as a text file'):
        tmp_download_link = download_link(comment, 'commend.txt', 'Click here to download comment text!')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

if __name__ == '__main__':
    main()