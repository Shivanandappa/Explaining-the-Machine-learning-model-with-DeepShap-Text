import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from tensorflow.keras.models import model_from_yaml
from streamlit_shap import st_shap
import shap
import re
import pickle

#title
st.title('Flights Sentiment Analysis project by JAY & PETRO')

#markdown text 
st.markdown("This dashboard is about visualisation flight review")

#sidebar top
st.sidebar.title('Sentiment analysis of Flights')
                 
# sidebar markdown 
st.sidebar.markdown("We can analyse passengers review from this application")

data = pd.read_csv('review.csv')


#checkbox to show data .. check box is true .. show the data
if st.checkbox("Show Data"):
    st.write(data.head(50)) # head count 50

    
#Radio buttons + Visualisation

select = st.sidebar.selectbox('Visulisation of Tweets',['Histogram','Pie chart'],key=1) # another widget another key

sentiment = data['airline_sentiment'].value_counts()
sentiment=pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})
st.markdown("### Sentiment count")
if select == "Histogram":
        fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500) #plotly library
        st.plotly_chart(fig)
else:
        fig = px.pie(sentiment, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)
               
    
#subheader with radiobuttons
st.sidebar.subheader('Tweets Analyser map')
tweets=st.sidebar.radio('Sentiment Type',('positive','negative','neutral')) # the data will take from the airline sentiment column 

st.markdown("Some Tweet review:")
# Comment section 
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])  
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])


# Slider side
st.sidebar.markdown('Time & location of tweets')
hr = st.sidebar.slider("Houe of the day",0,23)
data['Date'] = pd.to_datetime(data['tweet_created']) #Pandas dataframe for time 
hr_data = data[data['Date'].dt.hour == hr]

if not st.sidebar.checkbox("Hide",False, key='1'):
    st.markdown("### location of the tweets based on the hour of the day")
    st.markdown("%i tweets during %i:00 and %i:00" % (len(hr_data),hr,(hr+1)%24))
    st.map(hr_data)
    

#MULSELECT SLIDER
st.sidebar.subheader("Airline tweets by sentiment")
choice = st.sidebar.multiselect("Airlines", ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key = '0')  
if len(choice)>0:
    air_data=data[data.airline.isin(choice)]
    # facet_col = 'airline_sentiment'
    fig1 = px.histogram(air_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment',labels={'airline_sentiment':'tweets'}, height=600, width=800)
    st.plotly_chart(fig1)  
    
# Load the model for shap 
# load YAML and create model
yaml_file = open(r"C:\Users\Petro\Desktop\Visual Analytics\Visual_Analytics\model.yaml", 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(r"C:\Users\Petro\Desktop\Visual Analytics\Visual_Analytics\model.h5")

# Load in Data for Shap
sentences_test = pd.read_csv("sentences_test.csv")
sentences_test = sentences_test.squeeze()

# load vectorizer
vectorizer = pickle.load(open("vectorizer.pk", "rb"))

shap.initjs()

#load X_test
X_test= pd.read_csv("X_test.csv")
# method to make prediction for shap 
def make_predictions(X_batch_text):
    X_batch = vectorizer.transform(X_batch_text).toarray()
    preds = loaded_model.predict(X_batch)
    return preds


selected_categories=['negative','neutral','positive']
masker = shap.maskers.Text(tokenizer=r"\W+")
explainer = shap.Explainer(make_predictions, masker=masker, output_names=selected_categories)



X_batch_text = sentences_test[1:5]
X_batch = X_test[1:5]


shap_values = explainer(X_batch_text)


st_shap(shap.text_plot(shap_values),height=2000, width=1000)
