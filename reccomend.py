from fastapi import FastAPI, Depends, Request, Form, status
import joblib
import pandas as pd
import numpy as np
from starlette.responses import RedirectResponse
from starlette.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.orm import Session
import pandas as pd
from database import SessionLocal, engine
import models 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pymssql
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from typing import Optional


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db 
    finally:
        db.close()
        
# set some variables that will be used later
# these columns are the one to be scaled and given to the model 

filtered_columns = [
    'Urgent', 'Quantity', 'Shift', 'Machines_ID',
    'Machines_Zone', 'Capability', 'Snum_id', 'ypn1_id', 'ypn2_id', 'isFilAvail', 
    'Day', 'Month', 'Hour'
] 
  
    
# Load the model and the scaler 
model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')

#establish connection with the database taht contains the data for testing the model 
con = pymssql.connect(server='DESKTOP-IDFN0VR\SQL_SERVER_2024',user='sa',password='123456789',database='Yazaki_data')
cursor=con.cursor()





# here it starts the api
app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")





@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("home.html",
                                      {"request": request})


@app.post("/predict")
async def predict( a: int = Form(...),number_orders: int= Form(...),):
    X=get_test_data(a)
    print("here is original X",X)
    results=get_result(X)
    return get_top_snums(results,number_orders)


@app.get("/test_scaler")
async def test( ):
    X=get_test_data(5)
    print('original :\n',X)
    X=prepare_data(X,scaler)
    print('scaled x \n',X)
    X[filtered_columns]=scaler.inverse_transform(X[filtered_columns])
    print('original reversed \n',X)
    return 'look at the terminal'












# In this section you will find  functions that the api is using 



def get_test_data(a):
    # This function gets the data from the database and variable a is number of rows to be tested upon
    # NEWID() randomizese the extraction so each time we get differenet set

    query_table = f"""
    SELECT TOP {a} *
    FROM ymt_3
    ORDER BY NEWID();
    """
    
    # we use drop duplicate to get unique orders 
    df = pd.read_sql(query_table, con).drop_duplicates(subset='SNumber', keep='first')
    return df
    


def get_result(df):
    # this function calculates the probability of each order to be in pos 1 or 2 or 3 
    
    df=prepare_data(df,scaler)
    results = []  
    for index, row in df.iterrows():
        row = df.iloc[[index]]   #  the row is converted to a  dataframe beacsue the scaler only accepts datfarames
        snumber = row['SNumber'].item()
        input_vector  = row[filtered_columns].values.astype(float).reshape(1, -1)
        probabilities = model.predict(input_vector)  

        probabilities = probabilities[0]
        results.append({
            'SNumber': snumber,
            'Prob_1': float(probabilities[0]),  # probabilty for this order to be in seq 1 
            'Prob_2': float(probabilities[1]),  # probabilty for this order to be in seq 2
            'Prob_3': float(probabilities[2])   # probabilty for this order to be in seq 3
        })
    results_df = pd.DataFrame(results)
    return results_df
    


def get_top_snums(results, n):
    # this function sorts for each sequence (1,2,3) the orders that are more probabel to be in that sequence and it adds it to variable r
    # as you see  variable n sets the limit of orders to be confirmed n can be 1 ,2 ,3  and its given by the user in home page 
    # it returns a dictionnary where the key is snumber and the item is the probability to be in that sequence 
    # You can change the items and replace probabilty with direct sequence like {S221:1,sk234:2,s0002:3} by using the i in loop 
    r = {}
    
    results_dropped = results.copy()
    print(type(results))
    for i in range(1, n + 1):
        # Find the index of the maximum value for the current probability column
        max_index = results[f'Prob_{i}'].idxmax()
        row_to_drop = results.loc[max_index]

        # Extract the 'SNumber' from the row
        snumber = row_to_drop['SNumber']

        # Store the 'SNumber' and the corresponding probability in the dictionary
        r[snumber] = row_to_drop[f'Prob_{i}']  # replace  f'Prob_{i}'  by i if you want to get the sequence


        # Drop the row with the current 'SNumber' from the DataFrame
        results = results.drop(max_index)
        
        # Reset the index of the DataFrame
        results = results.reset_index(drop=True)
        
        # Debug print statements
        print(f"Filtered rows after dropping '{snumber}':")
        print(results)
        print("Current ranking dictionary:", r)

    return r







## DIFFERENT FUNCTION TO BE USED WHEN EXTRACTIN DATA FROM A    DATAFRAME 
def prepare_data(df,s):

    #Deal with null values 
    # Find duplicates based on all columns
    duplicate_rows = df[df.duplicated(keep=False)]
    df = df.drop_duplicates(keep='first')
    
    # Replace null values in the ypn2_id column with 0
    df['ypn2_id'] = df['ypn2_id'].fillna(0)
    df['ypn1_id'] = df['ypn1_id'].fillna(0)

    # Extract the numeric part from the Shift column
    df['Shift'] = df['Shift'].str.extract(r'(\d+)').astype(int)
    df['Machines_Zone'] = df['Machines_Zone'].str.extract(r'(\d+)').astype(int)
    
    # Replace 'sw' with 0 and 'dw' with 1 in the Capability column
    df['Capability'] = df['Capability'].replace({'SW': 0, 'DW': 1})
    
    # Convert Orderdate to datetime  
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    
    # Extract day, month, and hour from Orderdate to get non linear features
    df['Day'] = df['OrderDate'].dt.day
    df['Month'] = df['OrderDate'].dt.month
    df['Hour'] = df['OrderDate'].dt.hour

    #Scaling
    # Scale the remaining numerical  us transform function to keep compatible with training data 
    df[filtered_columns] = s.transform(df[filtered_columns])

    # Drop the specified columns
    df = df.drop(columns=['Seq','ProdProgress','YPN1', 'YPN2', 'ActionStatus','CAONO','ActionStatusDate','Shiftdate','OrderDate'])
    
    return df.copy()



