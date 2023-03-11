import streamlit as st
import pandas as pd
from PIL import Image

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
            with info_tab:
                st.header("*Profil client*")
                st.markdown('*In this section, you will find all the main descriptive information of the client.  In order to have with the credit score of said customer, \
                            please refer to the following tab (Score Credits (details)) or to the vertical bar on the right.*', unsafe_allow_html=True)
                st.write("---" * 40)
                st.markdown('  ')
            with credit_tab:
                # ----------------------------------------------------------
                # New subsection :
                st.header("*Scoring credit*")
                # ----------------------------------------------------------
                st.markdown('Summary of **important metrics**:<br>', unsafe_allow_html=True)
                # Display importante metrics : 
                n1, n2, n3 = st.columns((1,1,1))
                n1.metric(label ='Loans amount asked',value = 1)
                n2.metric(label ='Amount of total income',value = 20)
                n3.metric(label ='Amount Annuity',value = 3)

                st.write("---" * 40) # Add splitting line
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
                st.write("---" * 40) # Add splitting line
                # ----------------------------------------------------------
                # Graphic 2 : Biplot
                # ----------------------------------------------------------
                st.markdown('This second visual display allow to observe the distribution one numerical client parameters according to an another. \nPlus, this display can be colored by a specific categorie, allowing a better understanding of the client profil.<br>', unsafe_allow_html=True)
                st.write("---" * 40) # Add splitting line
            with feat_impt:
                # ----------------------------------------------------------
                # Graphic 3 : Global Model feature importance
                # ----------------------------------------------------------
                # Brief label name preparation
                # feature_importance["feature"] = feature_importance["feature"].str.replace("_"," ") # STANDBY
                st.header("*Global feature importance*")
                st.markdown("Here are display major features involving in the global credit scoring client classification (n.b. acceptance/rejection).",unsafe_allow_html=True)
                st.write("---" * 40) # Add splitting line  
                # ----------------------------------------------------------
                # Graphic 4 : Current Client feature importance
                # ----------------------------------------------------------
                st.header("*Local feature importance*")
                st.markdown("In this section, the **major feature** involved in credit acceptance or rejection are display **for the selected client**.<br> \
                        You will be able to see wich feature and the exact value mainly responsable for the credit status.<br>\
                        In blue are indicated feature influencing in credit acceptance and in orange the rejection.",unsafe_allow_html=True)


if __name__ == '__main__':
    main()