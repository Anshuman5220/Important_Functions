## Building script

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import wptools
import time
import re

## Defining Location of data
data_loc = '/home/jupyter/data/'
output_loc = '/home/jupyter/output/'
script_loc = '/home/jupyter/code/'
version="80K_All_v1"

## Reading 100K dataset 

df = pd.read_csv(output_loc + 'root_dishes_booking_100K.csv')

df = df.iloc[80000:105792]
print("Reading data from row 80000:100000")
df.columns = ['processed_name','order_per_day','total_dishes']
dishes = list(df['processed_name'].unique())
print("Length of dishes is : {0}".format(len(dishes)))

print("Starting the Loop for {0} dishes".format(len(dishes)))
wiki_data = []
# attributes of interest contained within the wiki infoboxes
features = ['caption','region'
'national_cuisine'
'course'
'served','main_ingredient'
'variations','type']
counter=0
for dish in dishes:
    counter = counter +1
    print("Total dishes processed {0}".format(counter))
    dish = dish.lower()
    try:
        page = wptools.page(dish) # create a page object

        page.get_query()
        page.get_wikidata()
        try:
            page.get_parse() # call the API and parse the data
            if page.data['infobox'] != None:
                # if infobox is present
                infobox = page.data['infobox']
                # get data for the interested features/attributes
                data = { feature : infobox[feature] if feature in infobox else '' 
                             for feature in features }
            else:
                data = { feature : '' for feature in features }

            data['dish_name'] = dish
            try:
                data['category'] = str(page.data['aliases'])
            except KeyError:
                data['category'] = ''

            try:
                data['description'] = str(page.data['description'])
            except KeyError:
                data['description'] = ''

            try:
                data['origin'] = str(page.data['wikidata'].get('country of origin (P495)'))
            except KeyError:
                data['origin'] = ''

            try:
                data['cusine'] = page.data['wikidata'].get('part of (P361)')
            except KeyError:
                data['cusine'] = ''

            try:
                data['subclass'] = str(page.data['wikidata'].get('subclass of (P279)'))
            except KeyError:
                data['subclass'] = ''
                
            wiki_data.append(data)

        except KeyError:
            pass
        if counter in [2000,4000,6000,8000,10000,12000,14000,16000,18000,20000]:
            print("Sleeping for 5 mins after iteration {0}".format(counter))
            time.sleep(300)
            
    except:
        print("{0} Dish Not Found".format(dish))
        pass


Processed_data = pd.DataFrame(wiki_data)
Processed_data = Processed_data[['dish_name','category','description','origin'
'cusine'
'subclass','main_ingredient'
'variations'
'type','caption'
'region'
'national_cuisine'
'course'
'served']]
Processed_data['region'] = Processed_data['region'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['national_cuisine'] = Processed_data['national_cuisine'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['course'] = Processed_data['course'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['main_ingredient'] = Processed_data['main_ingredient'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['caption'] = Processed_data['caption'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['category'] = Processed_data['category'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['origin'] = Processed_data['origin'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['variations'] = Processed_data['variations'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['type'] = Processed_data['type'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))
Processed_data['subclass'] = Processed_data['subclass'].apply(lambda x: re.sub(r"[\([{})\]]"
""
x))

print("Total of dishes captured from wikidata are : {0}".format(Processed_data.shape))
Processed_data.to_csv(output_loc+"RootDishes_100K_{0}.csv".format(version),index=False)


