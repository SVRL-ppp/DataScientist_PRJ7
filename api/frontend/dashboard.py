import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import pickle
import dill
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
from lime import lime_tabular
from streamlit import components
import xgboost
import os
import statsmodels # for docker 

# --------------------------------------------------------------------------------------------------------------
# API COMMUNICATION
# --------------------------------------------------------------------------------------------------------------
# interact with FastAPI endpoint
# "http://backend.docker:8000/predict"
endpoint = 'http://host.docker.internal:8000/predict' # Specify this path for Dockerization to work
# endpoint = 'http://localhost:8000/predict'
# --------------------------------------------------------------------------------------------------------------
# LOAD
# --------------------------------------------------------------------------------------------------------------
# api_path = '../data/'
api_path = os.path.split(os.getcwd())[0] + '/data/'

application_test = pd.read_csv(api_path + "application_test_df.csv")

with open(api_path + 'model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(api_path + 'feats.pkl', 'rb') as f:
    feats = pickle.load(f)

with open(api_path + 'BestModel_FeatureImportance.pkl', 'rb') as f:
    feature_importance = pickle.load(f)

# Load of the threshold value used for the best model
with open(api_path+'bestmodel_threshold.pkl', 'rb') as f:
    model_threshold = pickle.load(f)

df_test = pd.read_csv(api_path + "df_test_st_support.csv")

# for LIME explainer : 
# Load of best model explainer
with open(api_path + "LIME_explainer", 'rb') as in_strm:
    explainer = dill.load(in_strm)

# --------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------------------------------------------------
def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive/365
    return age_years

def inverseOHE(dataframe,pattern):
    list = dataframe.columns[dataframe.columns.str.contains(pattern) & ~dataframe.columns.str.contains("PREV")]
    cat = dataframe[list].idxmax(1).str.replace(pattern+"_","")
    return cat
# --------------------------------------------------------------------------------------------------------------
# METRIC BOX STYLE
# --------------------------------------------------------------------------------------------------------------
st.markdown("""<style>
div[data-testid="metric-container"] {background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: rgb(30, 103, 119);
   overflow-wrap: break-word;}
/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {overflow-wrap: break-word;
   white-space: break-spaces;
   color: cadetblue;}</style>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------------

def main():
    with st.spinner('Updating ...'):
        # ----------------------------------------------------------------------------------------
        # TITLE AND LOGO
        # ----------------------------------------------------------------------------------------
        col1, col2 = st.columns([1, 4])
        with col1:
            logo = Image.open('logo.png')
            st.image(logo,width=150,output_format='PNG')
        with col2:
            st.title('**Prêt à dépenser**: Client profil credit tool')

        # Tab creation
        info_tab, credit_tab, entities_tab, feat_impt = st.tabs(["Informations", "Score Credits (details)", "Related display","Features importances"])
        st.sidebar.title("Scoring prediction")

        with st.sidebar:
            # ----------------------------------------------------------------------------------------
            # ID CLIENT SELECTION 
            # ----------------------------------------------------------------------------------------
            client_choice = st.selectbox("Enter/Select a client number :",application_test['SK_ID_CURR'].unique())
            data_api = {'SKID': str(client_choice)}
            # Threshold have to be corrected too, because it was base on the probability that the client will no repay (probability of classe 1)
            model_threshold_corr = 1 - .4 # STANDBY TEMPORAIRE

            # ----------------------------------------------------------------------------------------
            # SCORING CLIENT PREDICTION
            # ----------------------------------------------------------------------------------------
            if st.button('Start Prediction'):
                with st.spinner('Prediction in Progress. Please Wait...'):
                    response = requests.post(endpoint, json=data_api)
                    client_score = response.json() # extraction of probability that the client will repay the loan (probability of classe 0)
                st.markdown("**Client score**: " + str(round(client_score['proba'],3)), unsafe_allow_html=True)

            # ----------------------------------------------------------------------------------------
            # CLIENT INFORMATION
            # ----------------------------------------------------------------------------------------
                with info_tab:
                    st.header("*Profil client*")
                    st.markdown('*In this section, you will find all the main descriptive information of the client.  In order to have with the credit score of said customer, \
                                please refer to the following tab (Score Credits (details)) or to the vertical bar on the right.*', unsafe_allow_html=True)
                    st.write("---" * 40)
                    st.markdown('  ')
                    age = convert_age(df_test["DAYS BIRTH"][df_test["SK ID CURR"] == client_choice])
                    st.markdown('**Age**: '+ str(int(age.values))+" years", unsafe_allow_html=True)
                    gender = df_test["CODE GENDER"].loc[df_test["SK ID CURR"] == client_choice].iloc[0]
                    st.markdown('**Gender**: ' + gender, unsafe_allow_html=True)
                    st.markdown("**Occupation type**: " + df_test["OCCUPATION TYPE"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.markdown('**Family status** : ' + df_test["NAME FAMILY STATUS"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.write("---" * 40)
                    st.markdown('**Client own a realty**: ' + df_test["FLAG OWN REALTY"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.markdown("**Client own a car**: " + df_test["FLAG OWN CAR"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.write("---" * 40)
                    st.markdown("**Housing type**: " + df_test["NAME HOUSING TYPE"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.markdown("**Organization type**: " + df_test["ORGANIZATION TYPE"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.markdown("**Education Type**: " + df_test["NAME EDUCATION TYPE"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.markdown("**Type Suite**: " + df_test["NAME TYPE SUITE"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    st.write("---" * 40)

                with credit_tab:
                    # ----------------------------------------------------------
                    # New subsection :
                    st.header("*Scoring credit*")
                    # ----------------------------------------------------------
                    st.markdown('Summary of **important metrics**:<br>', unsafe_allow_html=True)
                    # Display importante metrics : 
                    n1, n2, n3 = st.columns((1,1,1))
                    n1.metric(label ='Loans amount asked',value = df_test["AMT CREDIT"][df_test["SK ID CURR"] == client_choice])
                    n2.metric(label ='Amount of total income',value = df_test["AMT INCOME TOTAL"][df_test["SK ID CURR"] == client_choice])
                    n3.metric(label ='Amount Annuity',value = df_test["AMT ANNUITY"][df_test["SK ID CURR"] == client_choice])


                    st.markdown('**Contract Type**: ' + df_test["NAME CONTRACT TYPE"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
                    if float(client_score['proba'])<=model_threshold_corr:
                        st.markdown('Client **risk score**: <span style="color:#d92e2e">**'+ str(round(float(client_score['proba']),3)) + '**</span>',unsafe_allow_html=True)
                    else:
                        st.markdown('Client **risk score**: <span style="color:#24962f">**'+ str(round(float(client_score['proba']),3)) + '**</span>',unsafe_allow_html=True)

                    st.markdown('This client risk scoring is borned between **0 to 1**. <br>More the score is close to 1, the lower the financial risk associated with the loan. \
                        Conversely, more the risk score is close to 0, the greater the financial risk associated with the loan.<br>Below and equal to the risk threshold value (<span style="color:#e39c5d">orange</span> area), \
                        the request loan is rejected. Conversely, above the risk threshold value, the request loan is accepted.', unsafe_allow_html=True)
                    
                    dec = []
                    if float(client_score['proba'])<=model_threshold_corr:
                        dec = "The loan request is <b>refused</b>. The client will have difficulty to repay."
                    else:
                        dec = "The loan request is <b>accepted</b>. The client will repay the loan."
                    st.sidebar.markdown("**Decision**: " + dec, unsafe_allow_html=True)
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = float(client_score['proba']),
                        number = {'prefix': "Client score: ", 'font': {'size': 20}, 'suffix':"<br><br><br><b>Decision: " + dec},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "<b>Credit Score<br>",'font': {'size': 30}},
                        gauge = {'axis': {'range': [None, 1],'tickwidth': 2, 'tickcolor': "white",'tickfont' : {'size': 15}},
                                'bar': {'color': "cadetblue"},
                                'borderwidth': 1,
                                'bordercolor': "white",
                                'steps' : [
                                    {'range': [0, model_threshold_corr-.1], 'color': "rgba(255, 0, 0, 0.55)"},
                                    {'range': [model_threshold_corr-.1, model_threshold_corr], 'color': "rgba(255, 136, 0, 0.50)"},
                                    {'range': [model_threshold_corr, 1], 'color': "rgba(0, 255, 72, 0.28)"}],
                                'threshold' : {'line': {'color': "cadetblue", 'width': 3}, 'thickness': .50, 'value': float(client_score['proba'])}}))
                    fig.update_layout(font = {'color': "cadetblue"})
                    st.plotly_chart(fig, use_container_width=True)

                # ----------------------------------------------------------------------------------------
                # Fird tab : Plots
                # ----------------------------------------------------------------------------------------
                with entities_tab:
                    st.header("*Related display*")
                    st.subheader("*Visualization*")
                    # ----------------------------------------------------------
                    # Graphic 1 : features distribution 
                    # ----------------------------------------------------------
                    st.markdown('This first visual display allow to observe the distribution of all client parameters according to specific categories.<br>', unsafe_allow_html=True)
                    categorical_columns = [col for col in df_test.columns if df_test[col].dtype == 'object']
                    feature = st.selectbox("Feature :", [col for col in df_test.columns if col not in categorical_columns])
                    category = st.selectbox("Category :", [col for col in categorical_columns if col not in ["ORGANIZATION TYPE"]],key="CAT1") # We remove "ORGANIZATION TYPE" here 'cause the feature possess too much mode

                    fig = px.histogram(df_test, x=feature, color=category, marginal="box", nbins=20, title='Distribution of the parameters : <b>'+feature.replace("_"," ").capitalize()+\
                                    "<br><br><sup><span style='color:blue'>Categorical value for the current client: "+df_test[category].loc[df_test["SK ID CURR"] == client_choice].iloc[0]+"</sup>",\
                        opacity=0.2, color_discrete_sequence=px.colors.qualitative.Light24, barmode='overlay', width=800, height=400)

                    fig.update_layout(yaxis=dict(title_text="Count"),\
                        xaxis=dict(title_text=feature),\
                        titlefont=dict(size =18, color='black'),\
                        legend=dict(orientation="h", itemwidth=40, y=-.15, x=.5, xanchor="center", bordercolor="Black", borderwidth=.7, title_text="<b>"+category+"</b> :"))

                    fig.add_vline(x=float(df_test[feature].loc[df_test["SK ID CURR"] == client_choice].values), line_dash = 'dash', line_width=2, line_color = 'black')
                    # Fake plot for the legend    
                    fig.add_trace(go.Scatter(x=[df_test[feature].mean(), df_test[feature].mean()], y=[0,0], mode='lines', line=dict(color='black', width=2, dash='dash'), name='Current client positioning'))
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("---" * 40) # Add splitting line
                    # ----------------------------------------------------------
                    # Graphic 2 : Biplot
                    # ----------------------------------------------------------
                    st.markdown('This second visual display allow to observe the distribution one numerical client parameters according to an another. \nPlus, this display can be colored by a specific categorie, allowing a better understanding of the client profil.<br>', unsafe_allow_html=True)
                    feature1 = st.selectbox("Feature 1 :", [col for col in df_test.columns if col not in categorical_columns])
                    feature2 = st.selectbox("Feature 2 :", [col for col in df_test.columns if col not in categorical_columns])
                    category2 = st.selectbox("Category :", [col for col in categorical_columns if col not in ["ORGANIZATION TYPE"]], key="CAT2")

                    # Plot :
                    fig = px.scatter(df_test, x=feature1, y=feature2, color=category2, width=800, height=600, opacity=0.4,
                        trendline="ols", trendline_scope="overall",trendline_color_override="black",
                        color_discrete_sequence=px.colors.qualitative.Light24, title='<b>'+feature2+"</b> in function of <b>"+feature1+'</b> according to the <b>'+category2+
                            "<br><br><sup><span style='color:blue'>Categorical value for the current client: "+df_test[category2].loc[df_test["SK ID CURR"] == client_choice].iloc[0]+"</sup>")
                    fig.update_traces(marker={'size': 5})
                    fig.update_layout(yaxis=dict(title_text=feature2),
                        xaxis=dict(title_text=feature1),
                        titlefont=dict(size =18, color='black'),
                        legend=dict(orientation="h", itemwidth=40, y=-.15, x=.5,xanchor="center", bordercolor="Black", borderwidth=.7, title_text="<b>"+category+"</b> :"))
                    fig.add_trace(go.Scattergl(x=[float(df_test[feature1].loc[df_test["SK ID CURR"] == client_choice].values)],\
                        y=[float(df_test[feature2].loc[df_test["SK ID CURR"] == client_choice].values)],mode="markers",
                                    marker=dict(color="black", size=10, symbol="star"), name="Current client positioning"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("---" * 40)  # Add splitting line
                    
                with feat_impt:
                    # ----------------------------------------------------------
                    # Graphic 3 : Global Model feature importance
                    # ----------------------------------------------------------
                    # Brief label name preparation
                    feature_importance["feature"] = feature_importance["feature"].str.replace("_"," ")
                    st.header("*Global feature importance*")
                    st.markdown("Here are display major features involving in the global credit scoring client classification (n.b. acceptance/rejection).",unsafe_allow_html=True)

                    # Allowing the consellor to select the number of features 
                    # number = range(5,50,5) # Suppression du selectbox pour réduire le temps de chargement
                    number = 10
                    best_features = feature_importance[["feature", "importance"]].sort_values(by="importance", ascending=False)[:number]
                    set_color = px.colors.sample_colorscale("turbo", [n/(number -1) for n in range(number)])

                    fig = px.bar(best_features, y='feature', x='importance', height=600, width=800,\
                        hover_name="feature", title= "<b>Global features importances</b> : Top " + str(number) + " features",\
                        color='feature', color_discrete_sequence=set_color)
                    fig.update_layout(yaxis=dict(title_text="<b>Feature"),\
                        xaxis=dict(title_text="<b>Importance"),\
                        titlefont=dict(size =18, color='black'),
                        showlegend=False)

                    # to Prevent Axis Labels from being Cut Off :
                    fig.update_yaxes(automargin = True, # to Prevent Axis Labels from being Cut Off
                        ticksuffix = "  " # to add space between yaxis label and the plot
                        )
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("---" * 40) # Add splitting line  

                    # ----------------------------------------------------------
                    # Graphic 4 : Current Client feature importance
                    # ----------------------------------------------------------
                    st.header("*Local feature importance*")
                    st.markdown("In this section, the **major feature** involved in credit acceptance or rejection are display **for the selected client**.<br> \
                            You will be able to see wich feature and the exact value mainly responsable for the credit status.<br>\
                            In blue are indicated feature influencing in credit acceptance and in orange the rejection.",unsafe_allow_html=True)
                    id_numb = application_test[feats].loc[application_test['SK_ID_CURR']==client_choice].index[0]
                    explanation = explainer.explain_instance(np.array(application_test[feats])[id_numb], model.predict_proba, num_features=10)
                    html_lime=explanation.as_html()
                    components.v1.html(html_lime, width=1000, height=350, scrolling=True)
                    st.write("---" * 40) # Add splitting line

if __name__ == '__main__':
    main()