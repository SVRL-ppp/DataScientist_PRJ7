import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def convert_age(age_days_negative):
    age_days_positive = -age_days_negative
    age_years = age_days_positive/365
    return age_years

def inverseOHE(dataframe,pattern):
    list = dataframe.columns[dataframe.columns.str.contains(pattern) & ~dataframe.columns.str.contains("PREV")]
    cat = dataframe[list].idxmax(1).str.replace(pattern+"_","")
    return cat

# --------------------------------------------------------------------------------------------------------------
# SUPPORT LOAD
output_path = '/Users/sandrineveloso/Documents/COURS_ENSEIGNEMENT_FORMATION/FORMATIONS/2103 OPENSCLASSROOM/7_PROJECT7/LIVRABLES/DataScientist_PRJ7/output/'
path_models_variables = '/Users/sandrineveloso/Documents/COURS_ENSEIGNEMENT_FORMATION/FORMATIONS/2103 OPENSCLASSROOM/7_PROJECT7/LIVRABLES/DataScientist_PRJ7/models_variables/'
df_output = pd.read_csv(output_path + "submission_kernel.csv")
df_output.columns = df_output.columns.str.replace("_"," ")
df = pd.read_csv(output_path + "original_dataframe.csv")
df.columns = df.columns.str.replace("_"," ")

# Load of the threshold value used for the best model
with open(path_models_variables+'bestmodel_threshold.pkl', 'rb') as f:
    model_threshold = pickle.load(f)

# Load of list of categorical variable
with open(output_path + 'cat_feat_list.pkl', 'rb') as f:
    categorical_columns = pickle.load(f)

# Load the global model feature importance
with open(output_path + 'BestModel_FeatureImportance.pkl', 'rb') as f:
    feature_importance = pickle.load(f)

# --------------------------------------------------------------------------------------------------------------
# DATAFRAME PREPARATION
df_test = df[df['SK ID CURR'].isin(df_output['SK ID CURR'].tolist())]
df_test = pd.merge(df_test,df_output,on="SK ID CURR")
df_test.columns = df_test.columns.str.replace("_"," ")

df_test["STATUS"] = df_test["TARGET"]
df_test["STATUS"].loc[df_test["STATUS"] >= model_threshold] = 1
df_test["STATUS"].loc[df_test["STATUS"] < model_threshold] = 0
df_test["STATUS"] = df_test["STATUS"].astype(str).replace("1.0","Loan refused").replace("0.0","Loan accepted")

# Categorical variable : identification, cleaning and reencoding :
with open(output_path + 'cat_feat_list.pkl', 'rb') as f:
    categorical_columns = pickle.load(f)

pd.Series(categorical_columns).str.rsplit("_",n=2)# n=1
categorical_features = pd.Series(categorical_columns).str.rsplit('_', 1).str.get(0).str.replace("_"," ").unique()
categorical_features = [col for col in categorical_features if col not in ["NAME TYPE SUITE Other", "NAME YIELD GROUP low","CODE","FLAG OWN"]]

# Remove category only linked to aggregations so not re-encodable
rem = ["NAME PAYMENT TYPE","FLAG LAST APPL PER CONTRACT","NAME CASH LOAN PURPOSE","NAME CONTRACT STATUS","CODE REJECT REASON","NAME CLIENT TYPE","NAME GOODS CATEGORY","NAME PORTFOLIO","NAME PRODUCT TYPE","CHANNEL TYPE", 
"NAME SELLER INDUSTRY","NAME YIELD GROUP","PRODUCT COMBINATION"]
categorical_features = [col for col in categorical_features if col not in rem]

# Reencoding of categorial variables :
df_test['CODE GENDER'] = df_test['CODE GENDER'].apply(lambda x : "M" if x==0 else "F")
df_test['FLAG OWN CAR'] = df_test['FLAG OWN CAR'].apply(lambda x : "No" if x==0 else "Yes")
df_test['FLAG OWN REALTY'] = df_test['FLAG OWN REALTY'].apply(lambda x : "Yes" if x==0 else "No")

for i in range(len(categorical_features)):
    list = df_test.columns[df_test.columns.str.contains(categorical_features[i]) & ~df_test.columns.str.contains("PREV ") & ~df_test.columns.str.contains("POS ") \
                           & ~df_test.columns.str.contains("CC ") & ~df_test.columns.str.contains("XNA")]
    cat = df_test[list].idxmax(1).str.replace(categorical_features[i]+" ","")
    df_test[categorical_features[i]] = cat
    df_test.drop(list.tolist(),axis=1,inplace=True)

full_cat_list = categorical_features + ['CODE GENDER', 'FLAG OWN CAR', 'FLAG OWN REALTY','STATUS']

# Some correction of strings : 
df_test["NAME CONTRACT TYPE"] = df_test["NAME CONTRACT TYPE"].str.replace("Revolvingloans","Revolving loans")
df_test["NAME TYPE SUITE"] = df_test["NAME TYPE SUITE"].str.replace("Groupofpeople","Group of people").str.replace("Spousepartner","Spouse/partner").str.replace("Other B","Other").str.replace("Other A","Other")
df_test["NAME HOUSING TYPE"] = df_test["NAME HOUSING TYPE"].str.replace("apartment"," apartment").str.replace("Withparents"," With parents")
df_test["OCCUPATION TYPE"] = df_test["OCCUPATION TYPE"].str.replace("staff"," staff").str.replace("Realtyagents","Realty agents").str.replace("LowskillLaborers","Low skill Laborers").\
    str.replace("Highskilltech","High skill tech").str.replace("Privateservice","Private service").str.replace("Waitersbarmen","Waiters/barmen")
df_test['WEEKDAY APPR PROCESS START'] = df_test['WEEKDAY APPR PROCESS START'].str.capitalize()
df_test["NAME INCOME TYPE"] = df_test["NAME INCOME TYPE"].str.replace("Stateservant","State servant").str.replace("Commercialassociate","Commercial associate")
df_test["NAME FAMILY STATUS"] = df_test["NAME FAMILY STATUS"].str.replace("Singlenotmarried","Single/not married").str.replace("Civilmarriage","Civil marriage")
df_test["NAME EDUCATION TYPE"] = df_test["NAME EDUCATION TYPE"].str.replace("Highereducation","Higher education").str.replace("Secondarysecondaryspecial","Secondary/secondary special")\
    .str.replace("Incompletehigher","Incomplete higher").str.replace("Lowersecondary","Lower secondary").str.replace("Academicdegree","Academic degree")
df_test["FONDKAPREMONT MODE"] = df_test["FONDKAPREMONT MODE"].str.replace("notspecified","Not specified").str.replace("account"," account").str.capitalize()
df_test["HOUSETYPE MODE"] = df_test["HOUSETYPE MODE"].str.replace("blockofflats","Block of flats").str.replace("specifichousing","Specific housing").str.replace("terracedhouse","Terraced house")
df_test["ORGANIZATION TYPE"].loc[df_test["ORGANIZATION TYPE"].str.contains("Transporttype")] = "Transport type"
df_test["ORGANIZATION TYPE"].loc[df_test["ORGANIZATION TYPE"].str.contains("Industrytype")] = "Industry type"
df_test["ORGANIZATION TYPE"].loc[df_test["ORGANIZATION TYPE"].str.contains("Tradetype")] = "Trade type"
df_test["ORGANIZATION TYPE"].loc[df_test["ORGANIZATION TYPE"].str.contains("BusinessEntityType")] = "Business Entity Type"
df_test["ORGANIZATION TYPE"] = df_test["ORGANIZATION TYPE"].str.replace("Selfemployed","Self employed").str.replace("LegalServices","Legal Services").str.replace("SecurityMinistries","Security Ministries")

# --------------------------------------------------------------------------------------------------------------
# METRIC BOX STYLE
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
    MLFLOW_URI = 'http://127.0.0.1:5000'

    with st.spinner('Updating ...'):
        st.title('**Prêt à dépenser**: Client profil credit tool')
        info_tab, credit_tab, entities_tab, feat_impt = st.tabs(["Informations", "Score Credits (details)", "Related display","Features importances"])

        st.sidebar.title("Summary")
        # ----------------------------------------------------------------------------------------
        # CLIENT SELECTION
        # ----------------------------------------------------------------------------------------
        # ID_client= df_output['SK_ID°CURR'].unique() - STANDBY remove ?
        client_choice = st.sidebar.selectbox("Enter/Select a client number :",df_output['SK ID CURR'].unique())
        client_row = df_output["TARGET"][df_output["SK ID CURR"] == client_choice]

        # Modification of Score value, 1 - technical score ; because it's more logical for a non technical public
        client_row_corr = 1 - client_row.values
        # Attention with that modification, threshold have to be corrected too.
        model_threshold_corr = 1 - model_threshold

        # ----------------------------------------------------------------------------------------
        # Value preparation
        # ----------------------------------------------------------------------------------------
        age = convert_age(df_test["DAYS BIRTH"][df_test["SK ID CURR"] == client_choice])
        gender = df_test["CODE GENDER"].loc[df_test["SK ID CURR"] == client_choice].iloc[0]
 
                
        # ----------------------------------------------------------------------------------------
        # First tab : Profil client
        # ----------------------------------------------------------------------------------------
        with info_tab:
            st.header("*Profil client*")
            st.markdown('  ')
            st.markdown('**Age**: '+ str(int(age.values))+" years", unsafe_allow_html=True)
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

            # Important info are stored in sidebar
            st.sidebar.markdown('**Age**: '+ str(int(age.values))+" years", unsafe_allow_html=True)
            st.sidebar.markdown('**Gender**: ' + gender)
        
        # ----------------------------------------------------------------------------------------
        # Second tab : Scoring credit
        # ----------------------------------------------------------------------------------------
        with credit_tab:
            # ----------------------------------------------------------
            # New subsection :
            st.header("*Scoring credit*")
            # ----------------------------------------------------------

            # Display importante metrics : 
            n1, n2, n3 = st.columns((1,1,1))
            n1.metric(label ='Loans amount asked',value = df_test["AMT CREDIT"][df_test["SK ID CURR"] == client_choice])
            n2.metric(label ='Amount of total income',value = df_test["AMT INCOME TOTAL"][df_test["SK ID CURR"] == client_choice])
            n3.metric(label ='Amount Annuity',value = df_test["AMT ANNUITY"][df_test["SK ID CURR"] == client_choice])

            st.markdown('**Contract Type**: ' + df_test["NAME CONTRACT TYPE"].loc[df_test["SK ID CURR"] == client_choice].iloc[0], unsafe_allow_html=True)
            if client_row_corr>model_threshold_corr:
                st.markdown('Client **risk score**: <span style="color:#24962f">**'+ str(float(client_row_corr)) + '**</span>',unsafe_allow_html=True)
            else:
                st.markdown('Client **risk score**: <span style="color:#d92e2e">**'+ str(float(client_row_corr)) + '**</span>',unsafe_allow_html=True)

            st.markdown('This client risk scoring is borned between **0 to 1**. <br>More the score is close to 1, the lower the financial risk associated with the loan. \
                Conversely, more the risk score is close to 0, the greater the financial risk associated with the loan.<br>Below and equal to the risk threshold value (<span style="color:#e39c5d">orange</span> area), \
                the request loan is rejected. Conversely, above the risk threshold value, the request loan is accepted.', unsafe_allow_html=True)
            
            dec = []
            if client_row.values>=model_threshold:
                dec = "The loan request is <b>refused</b>. The client will have difficulty to repay."
            else:
                dec = "The loan request is <b>accepted</b>. The client will repay the loan."

            st.sidebar.markdown("**Decision**: " + dec, unsafe_allow_html=True)
            # st.markdown("<i class="fas fa-coins"></i>", unsafe_allow_html=True) STANDBY : https://fontawesome.com/v5/icons/coins?s=solid&f=classic
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = float(client_row_corr),
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
                        'threshold' : {'line': {'color': "cadetblue", 'width': 3}, 'thickness': .50, 'value': float(client_row_corr)}}))
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
            st.markdown('This first visual display allow to observe the distribution of all client parameters according to specific categories. \
                         We advise you particulary to observe the intresting distribution using the **STATUS** as categorical paramaters (Loans accepted or refused).<br>', unsafe_allow_html=True)
            feature = st.selectbox("Feature :", [col for col in df_test.columns if col not in full_cat_list])
            category = st.selectbox("Category :", [col for col in full_cat_list if col not in ["ORGANIZATION TYPE"]],key="CAT1") # We remove "ORGANIZATION TYPE" here 'cause the feature possess too much mode

            fig = px.histogram(df_test, x=feature, color=category, marginal="box", nbins=20, title='Distribution of the parameters : <b>'+feature.replace("_"," ").capitalize()+\
                               "<br><br><sup><span style='color:blue'>Categorical value for the current client: "+df_test[category].loc[df_test["SK ID CURR"] == client_choice].iloc[0]+"</sup>",\
                opacity=0.2, color_discrete_sequence=px.colors.qualitative.Light24, barmode='overlay', width=800, height=400)

            fig.update_layout(yaxis=dict(title_text="Count"),\
                xaxis=dict(title_text=feature),\
                titlefont=dict(size =18, color='black'),\
                legend=dict(orientation="h", itemwidth=40, y=-.15, x=.5, xanchor="center", bordercolor="Black", borderwidth=.7, title_text="<b>"+category+"</b> :"))

            fig.add_vline(x=float(df_test[feature].loc[df_test["SK ID CURR"] == client_choice].values), line_dash = 'dash', line_width=2, line_color = 'black')
            # Fake plot for the legend    
            fig.add_trace(go.Scatter(x=[df_test[feature].mean(), df_test[feature].mean()], y=[0,0], mode='lines', line=dict(color='black', width=2, dash='dash'), name='Current customer positioning'))
            st.plotly_chart(fig, use_container_width=True)

            # ----------------------------------------------------------
            # Graphic 1 bis : features distribution with categorical selection
            # ----------------------------------------------------------

            # Add splitting line
            st.write("---" * 40)
            # ----------------------------------------------------------
            # Graphic 2 : Biplot
            # ----------------------------------------------------------
            categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
            feature1 = st.selectbox("Feature 1 :", [col for col in df_test.columns if col not in full_cat_list])
            feature2 = st.selectbox("Feature 2 :", [col for col in df_test.columns if col not in full_cat_list])
            category2 = st.selectbox("Category :", [col for col in full_cat_list if col not in ["ORGANIZATION TYPE"]], key="CAT2")

            # Plot :
            fig = px.scatter(df_test, x=feature1, y=feature2, color=category2, width=800, height=600, opacity=0.4,\
                trendline="ols", trendline_scope="overall",trendline_color_override="black",\
                color_discrete_sequence=px.colors.qualitative.Light24, title='<b>'+feature2+"</b> in function of <b>"+feature1+'</b> according to the <b>'+category2+\
                    "<br><br><sup><span style='color:blue'>Categorical value for the current client: "+df_test[category2].loc[df_test["SK ID CURR"] == client_choice].iloc[0]+"</sup>")
            fig.update_traces(marker={'size': 5})
            fig.update_layout(yaxis=dict(title_text=feature2),\
                xaxis=dict(title_text=feature1),\
                titlefont=dict(size =18, color='black'),\
                legend=dict(orientation="h", itemwidth=40, y=-.15, x=.5,xanchor="center", bordercolor="Black", borderwidth=.7, title_text="<b>"+category+"</b> :"))
            fig.add_trace(go.Scattergl(x=[float(df_test[feature1].loc[df_test["SK ID CURR"] == client_choice].values)],\
                y=[float(df_test[feature2].loc[df_test["SK ID CURR"] == client_choice].values)],mode="markers",
                            marker=dict(color="black", size=10, symbol="star"), name="Current customer positioning"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Add splitting line
            st.write("---" * 40)

            
        with feat_impt:
            # ----------------------------------------------------------
            # Graphic 3 : Current Client feature importance
            # ----------------------------------------------------------
            

            st.write("---" * 40) # Add splitting line    
            # ----------------------------------------------------------
            # Graphic 4 : Global Model feature importance
            # ----------------------------------------------------------
            # Brief label name preparation
            feature_importance["feature"] = feature_importance["feature"].str.replace("_"," ")

            st.markdown("Major features involving in the global credit scoring client classification (n.b. accepted/refused).\
                Select the number of feature to display (n.b. the recommended value is **10 or 15**.)",unsafe_allow_html=True)

            # Allowing the consellor to select the number of features 
            number = range(5,50,5)
            number = st.selectbox("Number of feature to display:", number)

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
            
            # Add splitting line
            st.write("---" * 40)

if __name__ == '__main__':
    main()