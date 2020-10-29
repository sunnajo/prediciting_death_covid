import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pickle
import datetime as dt
import time

def main():
    # Navigation sidebar
    st.sidebar.header('Navigation')
    selection = st.sidebar.radio('Explore?', ['Home', 'Predictor', 'Data'])
    if selection == 'Home':
        home()
    elif selection == 'Predictor':
        classifier_page()
    elif selection == 'Data':
        # Functions to load data
        def load_raw_data(url):
            data = pd.read_csv(url)
            return data
        
        def format_cdc_data(df):
            date_cols = ['cdc_report_dt', 'pos_spec_dt', 'onset_dt']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d').dt.date
            return df

        def format_tracking_data(df):
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['pos_rate'] = df['positive']/df['totalTestResults']
            return df

        # Data sources
        st.sidebar.header('Interactive Data')
        data_source = st.sidebar.selectbox('Data Sources',
        ['Choose data source', 'CDC Public Use Surveillance Data', 'COVID Tracking Project'])
        
        # CDC data
        if data_source == 'CDC Public Use Surveillance Data':
            cdc_data_raw = load_raw_data('https://data.cdc.gov/resource/vbim-akqf.csv')
            cdc_data = format_cdc_data(cdc_data_raw)

            st.title('CDC Public Use Surveillance Data')
            st.markdown('https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/vbim-akqf', unsafe_allow_html=True)
            '\n'

            if st.checkbox('View raw data'):
                st.write(cdc_data_raw)
            '\n'

            st.header('Explore data')
            '\n'

            # Dictionary of variable names and corresponding column names for easy access
            vars_dict = {'Death': 'death_yn', 'Hospitalization': 'hosp_yn', 'ICU admission': 'icu_yn',
            'Medical condition': 'medcond_yn', 'Sex': 'sex', 'Age group': 'age_group', 'Race/Ethnicity': 'race_ethnicity_combined',
            'Current status': 'current_status'}
            vars_names = list(vars_dict.keys())

            vars_chosen = st.multiselect('Choose variables', vars_names)
            for var in vars_chosen:
                st.bar_chart(cdc_data[vars_dict[var]])
            '\n'
            
            # Separate dataframes
            death = cdc_data[cdc_data['death_yn'] == 'Yes']
            no_death = cdc_data[cdc_data['death_yn'] == 'No']

            st.markdown('*Compare patients who died to those who did not:*')
            '\n'
            vars_names2 = vars_names[1:]

            if st.checkbox('Death'):
                death_vars = st.multiselect('Choose variable', vars_names2)
                for var in death_vars:
                    st.bar_chart(death[vars_dict[var]])
            '\n'
            if st.checkbox('No death'):
                no_death_vars = st.multiselect('Choose variable ', vars_names2)
                for var in no_death_vars:
                    st.bar_chart(no_death[vars_dict[var]])

        # Tracking data
        elif data_source == 'COVID Tracking Project':
            
            # Load data
            current_data_raw = load_raw_data('https://api.covidtracking.com/v1/us/current.csv')
            current_data = format_tracking_data(current_data_raw)
            
            national_data_raw = load_raw_data('https://api.covidtracking.com/v1/us/daily.csv')
            national_data = format_tracking_data(national_data_raw)
            
            states_data_raw = load_raw_data('https://api.covidtracking.com/v1/states/daily.csv')
            states_data = format_tracking_data(states_data_raw)

            # Layout
            st.title('The COVID Tracking Project')
            st.markdown('https://covidtracking.com/data/national', unsafe_allow_html=True)
            '\n'
            st.header('Explore data')
            '\n'
            st.subheader('National data')
            '\n'

            ## Current data
            st.markdown('*Current data*')
            if st.checkbox('View raw current data'):
                st.write(current_data)
            '\n'
            today = dt.datetime.now().strftime('%m/%d/%Y')
            today_pos = current_data.loc[0,'positive']
            pos_incr = current_data.loc[0,'positiveIncrease']
            today_deaths = current_data.loc[0,'death']
            deaths_incr = current_data.loc[0,'deathIncrease']
            today_hosp = current_data.loc[0,'hospitalized']
            hosp_incr = current_data.loc[0,'hospitalizedIncrease']
            today_pos_rate = current_data.loc[0,'positive']/current_data.loc[0,'totalTestResults']
            today_icu = current_data.loc[0,'']
        
            # Area for displaying current data
            st.markdown('### As of **{}**'.format(today))
            '\n'
            st.write('Total positive cases: {} (+{})'.format(today_pos, pos_incr))
            st.write('Total deaths: {} (+{})'.format(today_deaths, deaths_incr))
            st.write('Total patients hospitalized: {} (+{})'.format(today_hosp, hosp_incr))
            st.write('Total patients in ICU: {}'.format(today_icu))
            st.write('Current positive rate: {:.3f}'.format(today_pos_rate))
            '\n'
            '\n'

            ## All time data
            # Functions for plotting variables
            def plot_national_var(var):
                # Create dataframe
                df = pd.concat([national_data['date'], national_data[var]], axis=1)
                df.sort_values(by='date', inplace=True)
                # Plot
                st.line_chart(df.rename(columns={'date': 'index'}).set_index('index'))

            # Variables
            national_dict = {'Total positive cases': 'positive',
            'Total deaths': 'death',
            'Total negative cases': 'negative',
            'Total hospitalized': 'hospitalizedCumulative',
            'Total in ICU': 'inIcuCumulative',
            'Total on ventilator': 'onVentilatorCumulative',
            'Total recovered': 'recovered',
            'Total test results': 'totalTestResults',
            'Overall positive rate': 'pos_rate',
            'Increase in positive cases': 'positiveIncrease',
            'Increase in deaths': 'deathIncrease',
            'Increase in hospitalized': 'hospitalizedIncrease',
            'Increase in negative cases': 'negativeIncrease',
            'Increase in total test results': 'totalTestResultsIncrease'}
            national_vars = list(national_dict.keys())

            st.markdown('*All data*')

            if st.checkbox('View raw national data'):
                st.write(national_data)

            national_vars_chosen = st.multiselect('Choose variable', national_vars)
            for var in national_vars_chosen:
                plot_national_var(national_dict[var])
            '\n'

def home():
    # Title page
    st.title('Can we predict the outcome of a case of COVID-19?')
    st.image('/Users/sunnajo/downloads/covidinfodemic.jpg', column_width=True)
    st.text('Image source: Getty Images, via yalemedicine.org')
    '\n'
    '\n'
    st.markdown('*Disclaimer*')
    st.write('This content is purely for educational purposes and should NOT be transmitted, used to guide clinical decision making and/or personal decisions regarding seeking medical care or treatment, and/or for any other real-world applications.')

# Load the model
pickle_in = open('classifier2.pkl', 'rb')
classifier = pickle.load(pickle_in)

# Defining the function that will make the prediction using the data which the user inputs
def prediction(icu, hosp, age_group, med_cond, Male, pos_rate):
    input = np.array([[icu, hosp, age_group, med_cond, Male, pos_rate]]).astype(np.float64)
    prediction = classifier.predict(input)
    return int(prediction)

def predict_prob(icu, hosp, age_group, med_cond, Male, pos_rate):
    input = np.array([[icu, hosp, age_group, med_cond, Male, pos_rate]]).astype(np.float64)
    prob = classifier.predict_proba(input)
    return np.array(prob)

def classifier_page():
    # Title
    st.title('Predicting the Outcome of a Patient with COVID-19')
    st.header('A Machine Learning Approach')
    '\n'
    st.image('/Users/sunnajo/downloads/covidml.jpeg')
    st.text('Image source: TABIP')
    '\n'
    '\n'
    st.markdown('*Disclaimer*')
    st.write('This content is purely for educational purposes and should NOT be transmitted, used to guide clinical decision making and/or personal decisions regarding seeking medical care or treatment, and/or for any other real-world applications.')
    '\n'
    '\n'
    
    # Functions
    def load_data(url):
        data = pd.read_csv(url)
        return data

    ## User input areas
    # Dictionary of age groups
    age_dict = {"0-9 years": 0, "10-19 years": 1, "20-29 years": 2, "30-39 years": 3,
    "40-49 years": 4, "50-59 years": 5, "60-69 years": 6, "70-79 years": 7, "80+ years": 8}
    age_list = list(age_dict.keys())

    st.markdown('### **How old is the patient?**')
    input_age = st.select_slider('', age_list)
    age_group = age_dict[input_age]
    '\n'
    
    st.markdown('### **Is the patient hospitalized?**')
    hosp = st.radio('', ["No", "Yes"])
    if hosp == "Yes":
        hosp = 1
    elif hosp == "No":
        hosp = 0
    '\n'
    
    st.markdown('### **Is the patient in the ICU?**')
    icu = st.radio('  ', ["No", "Yes"])
    if icu == "Yes":
        icu = 1
    elif icu == "No":
        icu = 0
    '\n'

    st.markdown('### **Does the patient have an underlying medical condition?**')
    med_cond = st.radio('   ', ["No", "Yes"])
    if med_cond == "Yes":
        med_cond = 1
    elif med_cond == "No":
        med_cond = 0
    '\n'

    st.markdown('### **What is the current positivity rate? (as a percentage)**')
    pos_rate = st.number_input('    ')
    if st.button("Look it up"):
        current_data = load_data('https://api.covidtracking.com/v1/us/current.csv')
        pos_rate_pct = float(current_data['positive']/current_data['totalTestResults'])*100
        st.write('{:.2f}%'.format(pos_rate_pct))
    '\n'

    st.markdown("### **What is the patient's biological sex?**")
    sex = st.radio('     ', ["Female", "Male", "Other"])
    if sex == "Male":
        Male = 1
    else:
        Male = 0
    '\n'
    '\n'
    '\n'

    # Prediction
    if st.button("Predict"):
        result = prediction(icu, hosp, age_group, med_cond, Male, pos_rate)
        prob_pct = (float(predict_prob(icu, hosp, age_group, med_cond, Male, pos_rate)[:,1]))*100
        '\n'
        if result == 0:
            st.success("The patient likely has a low risk of death")
        elif result == 1:
            st.warning("The patient has a higher risk of death")
        '\n'
        
        # Pause
        time.sleep(1)

        # Cue for navigating to data section
        st.markdown('### *How did we come up with this algorithm?*')
        '\n'
        st.subheader('Click on the sidebar for data sources')

    
if __name__=='__main__': 
    main()