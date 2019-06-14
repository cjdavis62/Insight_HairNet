from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

pd.options.display.max_columns=25


#Initialize app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



#After submitting request, serve up a page with the results
@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():

    hair_type = ""
    # Get the data for the user's hair color
    if request.form['hair_type'] == 'hair_blonde':
        hair_type = "blonde"
    if request.form['hair_type'] == 'hair_brunette':
        hair_type = "brunette"

    # Read in data from the database for hair types
    f=open("db.txt", "r")
    contents=0
    if not(contents):
        contents = f.read()
    username = contents.split("\n")[0]
    password = contents.split("\n")[1]
    dbname = 'photos_db'

    con = psycopg2.connect(database = dbname, user = username, password = password, port=5432, host= "/var/run/postgresql/")

    sql_query = """
    SELECT * FROM salon_data_table;
    """
    salon_data_from_sql = pd.read_sql_query(sql_query,con)
    salon_hair = salon_data_from_sql[salon_data_from_sql.prediction == hair_type]

    salon_hair_sorted = salon_hair.sort_values(by='confidence', ascending=False)

    salon_hair_photos_0_path = "/static/img/ritualsalonatx/" + salon_hair_sorted.iloc[0][1]
    salon_hair_photos_0_confidence = salon_hair_sorted.iloc[0][5]
    print(salon_hair_photos_0_confidence)

    # Get the salon name requested:
    salon_name = request.form['salon_name']
    print(salon_name)


    reviews_df = pd.read_csv("/home/cjdavis/insight/flaskapp/static/data/reviews_with_vader_sentiment.csv")
    salon_reviews = reviews_df[reviews_df.Title == salon_name]
    random_subset = salon_reviews.sample(n=2)
    print(random_subset.head())
    sentimentA = random_subset.iloc[0][1]
    reviewA = random_subset.iloc[0][2]
    sentimentB = random_subset.iloc[1][1]
    reviewB = random_subset.iloc[1][2]
    if (sentimentA == 0):
        sentimentA = "negative"
    else:
        sentimentA = "positive"
    if (sentimentB == 0):
        sentimentB = "negative"
    else:
        sentimentB = "positive"

    #return render_template('recommendations.html', hair_type)
    return render_template('recommendations.html', hair_type = hair_type, salon_photo_path = salon_hair_photos_0_path, salon_photo_confidence = salon_hair_photos_0_confidence,
                           sentimentA = sentimentA, sentimentB = sentimentB, reviewA = reviewA, reviewB = reviewB, salon = salon_name)


if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=5000, debug=True)
