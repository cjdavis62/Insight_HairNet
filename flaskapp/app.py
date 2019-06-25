from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

from yelp_functions import *

pd.options.display.max_columns=25


#Initialize app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



#After submitting request, serve up a page with the results
@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    error_flag = 0 # value for different errors
    # First deal with the sentiment checker to pick out the right salon! #

    ## Read in data from the database for hair types ##
    f=open("db.txt", "r")
    contents=0
    if not(contents):
        contents = f.read()
    username = contents.split("\n")[0]
    password = contents.split("\n")[1]
    dbname_reviews = 'reviews_db'

    con = psycopg2.connect(database = dbname_reviews, user = username, password = password, port=5432, host= "/var/run/postgresql/")
    sql_query_reviews = """
    SELECT * FROM reviews_data_table;
    """
    reviews_data_from_sql = pd.read_sql_query(sql_query_reviews,con)
    con.close()

    keyword = request.form['product'] # read in the request from the previous page

    # clean things up!
    reviews_data_from_sql = reviews_data_from_sql.drop(axis = 1, columns=["index"])

    # find out how often the value comes up!
    reviews_data_from_sql["has_keyword"], reviews_data_from_sql["sentence_list"], reviews_data_from_sql["average_sentiment_sentence"] = zip(*reviews_data_from_sql.apply(parse_for_word, keyword = keyword, axis=1))
    reviews_data_from_sql = reviews_data_from_sql[reviews_data_from_sql.has_keyword != 0]
    if reviews_data_from_sql.empty:
        print ("error: no entries in reviews")
        error_flag = 1

    # reshape the values to sum+average over everything
    d = {'has_keyword':'has_keyword_sum', 'average_sentiment_sentence':'average_sentiment_sentence_average',
     'sentiment_vader':'sentiment_vader_average', 'sentence_list':'sentence_list_combined',
     'Review': 'Review_sum'}
    sorted_mean_by_Title=reviews_data_from_sql.groupby('Title', as_index = False).agg({'has_keyword':'sum',
                                                                'average_sentiment_sentence':'mean',
                                                                'sentiment_vader':'mean',
                                                                'sentence_list':'sum',
                                                                'Review':'sum'}).rename(columns=d)
    sorted_mean_by_Title.sort_values(by=['has_keyword_sum'], ascending=False, inplace=True)

    # Get the final scores and sort values
    sorted_mean_by_Title['final_score'] = sorted_mean_by_Title.apply(scaled_combined_score, axis=1)
    sorted_review_data = sorted_mean_by_Title.sort_values(by=['final_score'], ascending=False)

    # Get the two highest salons!
    salon_highest = sorted_review_data.iloc[0][0]
    salon_second_highest = sorted_review_data.iloc[1][0]
    salon_third_highest = sorted_review_data.iloc[2][0]

    salon_highest_reviews = sorted_review_data.iloc[0][4]
    salon_second_highest_reviews = sorted_review_data.iloc[1][4]
    salon_third_highest_reviews = sorted_review_data.iloc[2][4]

    salon_highest_score = "{0:0.1f}".format(sorted_review_data.iloc[0][6] * 100.0)
    salon_second_highest_score = "{0:0.1f}".format(sorted_review_data.iloc[1][6] * 100.0)
    salon_third_highest_score = "{0:0.1f}".format(sorted_review_data.iloc[2][6] * 100.0)

    # combine the data to something useful
    salon_highest_data = [salon_highest, salon_highest_reviews, salon_highest_score]
    salon_second_highest_data = [salon_second_highest, salon_second_highest_reviews, salon_second_highest_score]
    salon_third_highest_data = [salon_third_highest, salon_third_highest_reviews, salon_third_highest_score]


    # Get more data from the salons #
    dbname_salons = 'salons_db'

    con = psycopg2.connect(database = dbname_salons, user = username, password = password, port=5432, host= "/var/run/postgresql/")
    sql_query_salons = """
    SELECT * FROM salons_data_table;
    """
    salons_data_from_sql = pd.read_sql_query(sql_query_salons,con)
    con.close()
    salons_data_from_sql = salons_data_from_sql.drop(axis = 1, columns=["index"])

    # Find the rows matching the salon
    salon_highest_df = salons_data_from_sql[salons_data_from_sql.Title == salon_highest]
    salon_second_highest_df = salons_data_from_sql[salons_data_from_sql.Title == salon_second_highest]
    salon_third_highest_df = salons_data_from_sql[salons_data_from_sql.Title == salon_third_highest]

    # fix up the salon data before moving on
    salon_highest_df.fillna("N/A", inplace = True)
    salon_second_highest_df.fillna("N/A", inplace = True)
    salon_third_highest_df.fillna("N/A", inplace = True)


    # In case multiple places with the salon, pick the one with the highest number of reviews #
    salon_highest_df.sort_values(by=['Number_of_reviews'], ascending=False, inplace=True)
    salon_second_highest_df.sort_values(by=['Number_of_reviews'], ascending=False, inplace=True)
    salon_third_highest_df.sort_values(by=['Number_of_reviews'], ascending=False, inplace=True)

    salon_highest_address = salon_highest_df.iloc[0][2]
    salon_highest_yelp_rating = salon_highest_df.iloc[0][3]
    salon_highest_services = salon_highest_df.iloc[0][5]
    if salon_highest_services == "":
        show_highest_services = 0
    else:
        show_highest_services = 1

    salon_second_highest_address = salon_second_highest_df.iloc[0][2]
    salon_second_highest_yelp_rating = salon_second_highest_df.iloc[0][3]
    salon_second_highest_services = salon_second_highest_df.iloc[0][5]
    if salon_second_highest_services == "":
        show_second_highest_services = 0
    else:
        show_second_highest_services = 1

    salon_third_highest_address = salon_third_highest_df.iloc[0][2]
    salon_third_highest_yelp_rating = salon_third_highest_df.iloc[0][3]
    salon_third_highest_services = salon_third_highest_df.iloc[0][5]
    if salon_third_highest_services == "":
        show_third_highest_services = 0
    else:
        show_third_highest_services = 1

    salon_highest_data.extend((salon_highest_address, salon_highest_yelp_rating, salon_highest_services))
    salon_second_highest_data.extend((salon_second_highest_address, salon_second_highest_yelp_rating, salon_second_highest_services))
    salon_third_highest_data.extend((salon_third_highest_address, salon_third_highest_yelp_rating, salon_third_highest_services))


    ##### hair stuffs ######

    # Get the user's hair color
    hair_type = request.form['hair_type']

    # Read in data from the database for hair types
    dbname_photos = 'photos_db'

    con = psycopg2.connect(database = dbname_photos, user = username, password = password, port=5432, host= "/var/run/postgresql/")
    sql_query_photos = """
    SELECT * FROM photos_data_table;
    """
    insta_data_from_sql = pd.read_sql_query(sql_query_photos,con)
    con.close()

    insta_data_from_sql.drop(columns=["index"], inplace=True)

    insta_data_from_sql = insta_data_from_sql[insta_data_from_sql.prediction == hair_type]

    # Find the highest scoring salons with instagram accounts
    list_of_instagram_names=["urbanbettysalon", "methodhair", "redstellasalon", "topazsalonaustin", "garboasalon", "frenchysbeauty",
                             "blackorchidsalon", "cnnhairteam", "acessalon", "benjaminbeausalon", "vainaustin", "loveandrootssalon", "wildorchidatx",
                             "salonsovayatx", "thesalonatthedomain", "ritualsalonatx", "bellasalonatx", "milkandhoneysalon", "pathsalon", "waterstone_salon"]
    list_of_instagram_names.extend(("wetsalon", "l7salon",
                           "shagaustin", "karusalonatx",
                           "salon_vela", "spoletisalon",
                           "birdsbarbershop", 'massagesway',
                           'floyds99barbershop', 'ruiz_salon',
                           'dolcesalonaustin', 'tanyafarishair',
                           'urbanhairatx'))

    list_of_salons_titles_from_insta = ['Urban Betty', 'Method.Hair', 'Red Stella Hair Salon', 'Topaz Salon', 'Garbo A Salon and Spa',
                                        "Frenchy's Beauty Parlor", 'Black Orchid Salon', 'CNN Hair Team Salon', 'Chuck Edwards The Salon',
                                        'Benjamin Beau Salon', 'Vain', 'Love + Roots', 'Wild Orchid Salon', 'Salon Sovay', 'The Salon at The Domain',
                                        'Ritual Salon', 'Bella Salon', 'SALON by milk + honey', 'Path Salon', 'WaterStone Salon']
    list_of_salons_titles_from_insta.extend(("Wet Salon and Studio", 'L7 Salon', "Shag Hair Salon", "Karu Salon", 'Salon Vela', 'Spoleti Salon',
                                             'Birds Barbershop', 'Massage Sway', "Floyd's 99 Barbershop", 'Ruiz Salon', 'Dolce',
                                             'Hair by Tanya Faris', "Urban Hair"))


    # Get the salons that have instagram accounts that I've been able to find
    sorted_review_data["in_insta"] = sorted_review_data.Title.isin(list_of_salons_titles_from_insta)
    salon_data_with_insta_df = sorted_review_data[sorted_review_data.in_insta == True]

    merged_salon_insta = pd.merge(insta_data_from_sql, salon_data_with_insta_df, left_on='salon_name', right_on='Title')
    merged_salon_insta_hair_type = merged_salon_insta[merged_salon_insta.prediction == hair_type]

    sorted_merged_salon_insta_hair_type = merged_salon_insta_hair_type.sort_values(by=['final_score'], ascending=False)
    highest_salon_insta = sorted_merged_salon_insta_hair_type.salon_name.unique()[0]
    second_highest_salon_insta = sorted_merged_salon_insta_hair_type.salon_name.unique()[1]
    third_highest_salon_insta = sorted_merged_salon_insta_hair_type.salon_name.unique()[1]


    sorted_merged_salon_insta_hair_type_highest = sorted_merged_salon_insta_hair_type[sorted_merged_salon_insta_hair_type.salon_name == highest_salon_insta]
    sorted_merged_salon_insta_hair_type_second_highest = sorted_merged_salon_insta_hair_type[sorted_merged_salon_insta_hair_type.salon_name == second_highest_salon_insta]
    sorted_merged_salon_insta_hair_type_third_highest = sorted_merged_salon_insta_hair_type[sorted_merged_salon_insta_hair_type.salon_name == third_highest_salon_insta]


    confidence_ranked_merged_salon_insta_hair_type_highest = sorted_merged_salon_insta_hair_type_highest.sort_values(by=['confidence'], ascending=False)
    confidence_ranked_merged_salon_insta_hair_type_second_highest = sorted_merged_salon_insta_hair_type_second_highest.sort_values(by=['confidence'], ascending=False)
    confidence_ranked_merged_salon_insta_hair_type_third_highest = sorted_merged_salon_insta_hair_type_third_highest.sort_values(by=['confidence'], ascending=False)


    salon_hair_photos_highest_data = []
    salon_hair_photos_second_highest_data = []
    salon_hair_photos_third_highest_data = []


    salon_hair_photos_highest_path = "/static/img/" + confidence_ranked_merged_salon_insta_hair_type_highest.iloc[0][2] + "/" + confidence_ranked_merged_salon_insta_hair_type_highest.iloc[0][0]
    salon_hair_photos_highest_confidence = confidence_ranked_merged_salon_insta_hair_type_highest.iloc[0][6]
    salon_hair_photos_highest_insta_name = confidence_ranked_merged_salon_insta_hair_type_highest.iloc[0][2]
    salon_hair_photos_highest_name = confidence_ranked_merged_salon_insta_hair_type_highest.iloc[0][3]
    salon_hair_photos_highest_score = "{0:0.1f}".format(confidence_ranked_merged_salon_insta_hair_type_highest.iloc[0][13] * 100.0)

    salon_hair_photos_highest_data.extend((salon_hair_photos_highest_path, salon_hair_photos_highest_confidence,
                                           salon_hair_photos_highest_insta_name, salon_hair_photos_highest_name,
                                           salon_hair_photos_highest_score))



    salon_hair_photos_second_highest_path = "/static/img/" + confidence_ranked_merged_salon_insta_hair_type_second_highest.iloc[0][2] + "/" + confidence_ranked_merged_salon_insta_hair_type_second_highest.iloc[0][0]
    salon_hair_photos_second_highest_confidence = confidence_ranked_merged_salon_insta_hair_type_second_highest.iloc[0][6]
    salon_hair_photos_second_highest_insta_name = confidence_ranked_merged_salon_insta_hair_type_second_highest.iloc[0][2]
    salon_hair_photos_second_highest_name = confidence_ranked_merged_salon_insta_hair_type_second_highest.iloc[0][3]
    salon_hair_photos_second_highest_score = "{0:0.1f}".format(confidence_ranked_merged_salon_insta_hair_type_second_highest.iloc[0][13] * 100.0)

    salon_hair_photos_second_highest_data.extend((salon_hair_photos_second_highest_path, salon_hair_photos_second_highest_confidence, salon_hair_photos_second_highest_insta_name, salon_hair_photos_second_highest_name))


    salon_hair_photos_third_highest_path = "/static/img/" + confidence_ranked_merged_salon_insta_hair_type_third_highest.iloc[0][2] + "/" + confidence_ranked_merged_salon_insta_hair_type_second_highest.iloc[0][0]
    salon_hair_photos_third_highest_confidence = confidence_ranked_merged_salon_insta_hair_type_third_highest.iloc[0][6]
    salon_hair_photos_third_highest_insta_name = confidence_ranked_merged_salon_insta_hair_type_third_highest.iloc[0][2]
    salon_hair_photos_third_highest_name = confidence_ranked_merged_salon_insta_hair_type_third_highest.iloc[0][3]
    salon_hair_photos_third_highest_score = "{0:0.1f}".format(confidence_ranked_merged_salon_insta_hair_type_third_highest.iloc[0][13] * 100.0)

    salon_hair_photos_third_highest_data.extend((salon_hair_photos_third_highest_path, salon_hair_photos_third_highest_confidence, salon_hair_photos_third_highest_insta_name, salon_hair_photos_third_highest_name))



    '''
    # now combine the three sets of information! This has a few different scenarios.
    # case 1: If the first photo set matches the first salon set, then all 3 match one-to-one
    # case 2: If the first photo set matches the second salon set, then just 2 match
    # case 3: If the first photo set matches the third salon set, then only one matches
    # case 4: If none of the photo sets match, then we've got a problem!!
    case = 0
    if (salon_hair_photos_highest_name == salon_highest):
        case = 1
    elif (salon_hair_photos_highest_name == salon_second_highest):
        case = 2
    elif (salon_hair_photos_highest_name == salon_third_highest):
        case = 3
    else:
        case = 4

    # Do stuff for each of these cases
    show_photo_data = [0,0,0]
    if (case == 1):
        show_photo_data = [1,1,1]
        salon_highest_data.extend(salon_hair_photos_highest_data)
        salon_second_highest_data.extend(salon_hair_photos_second_highest_data)
        salon_third_highest_data.extend(salon_hair_photos_third_highest_data)
    if (case == 2):
        show_photo_data = [0,1,1]
        salon_second_highest_data.extend(salon_hair_photos_highest_data)
        salon_third_highest_data.extend(salon_hair_photos_second_highest_data)
    if (case == 3):
        show_photo_data = [0,0,1]
        salon_third_highest_data.extend(salon_hair_photos_highest_data)
        '''

    ### Fix to cases ###
    show_photo_data = [0,0,0]

    salon_names = [salon_highest, salon_second_highest, salon_third_highest]
    salon_photos_names = [salon_hair_photos_highest_name, salon_hair_photos_second_highest_name, salon_hair_photos_third_highest_name]
    salon_datas = [salon_highest_data, salon_second_highest_data, salon_third_highest_data]
    salon_photos_datas = [salon_hair_photos_highest_data, salon_hair_photos_second_highest_data, salon_hair_photos_third_highest_data]

    for i in range(len(salon_names)): # look through all the salon names
        for j in range(len(salon_photos_names)): #look through all the salon names for the instagram photo
            if (salon_names[i] == salon_photos_names[j]): # check if names are the same
                show_photo_data[i] = 1 # if they're the same, then yes, show this photo
                salon_datas[i].extend((salon_photos_datas[j]))

    #### Nearly done!!! ####
    # To deal with case 0, where things are bad and there are no photos, have to rely on other photos to supplement!
    # Look at all the photos that satisfy the color requirement, and take those to show, based on rankings

    sorted_merged_salon_insta_good_score = sorted_merged_salon_insta_hair_type[sorted_merged_salon_insta_hair_type.confidence >= 0.8]
    end_photo_list = []
    min_photos = 10
    # make more photos if in case 4 because that is when life is bad
    if (show_photo_data == [0, 0, 0]):
        min_photos = 15
    number_of_photos = min(min_photos, sorted_merged_salon_insta_good_score.shape[0])
    indices = np.random.choice(sorted_merged_salon_insta_good_score.shape[0], size = number_of_photos, replace = False)
    for index in indices:
        photo_information = []
        photo_name = sorted_merged_salon_insta_good_score.iloc[index][0]
        photo_location = "/static/img/" + sorted_merged_salon_insta_good_score.iloc[index][2] + "/" + photo_name
        instagram_name = sorted_merged_salon_insta_good_score.iloc[index][2]
        salon_name = sorted_merged_salon_insta_good_score.iloc[index][3]
        score = "{0:0.1f}".format(sorted_merged_salon_insta_good_score.iloc[index][13] * 100.0)
        photo_information.extend((photo_location, instagram_name, salon_name, score))
        end_photo_list.append(photo_information)
    # sort the list properly
    def takeFourth(elem):
        return elem[3]
    end_photo_list.sort(key=takeFourth, reverse=True)

    #return the next page for the next step
    return render_template('split_recommendation_flexbox.html', hair_type = hair_type, keyword = keyword,
                           salon_highest_data = salon_highest_data, salon_second_highest_data = salon_second_highest_data, salon_third_highest_data = salon_third_highest_data,
                           show_photo_data = show_photo_data, show_highest_services = show_highest_services, show_second_highest_services = show_second_highest_services,
                           end_photo_list = end_photo_list)


if __name__ == '__main__':
    #this runs your app locally
    app.run(host='0.0.0.0', port=5000, debug=True)
