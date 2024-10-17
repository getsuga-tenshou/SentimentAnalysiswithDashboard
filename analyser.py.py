import os
from google.cloud import bigquery
import pandas_gbq
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score

def initialize_ds():

    sql_query = """
    SELECT 
        GLOBALEVENTID, Actor1Geo_FullName, Actor2Geo_FullName, ActionGeo_FullName,
        NumSources, NumMentions, NumArticles, QuadClass, GoldsteinScale, AvgTone, DATEADDED
    FROM 
        `gdelt-bq.gdeltv2.events`
    WHERE
        Actor1Geo_FullName IS NOT NULL 
        AND Actor2Geo_FullName IS NOT NULL 
        AND ActionGeo_FullName IS NOT NULL
        AND (QuadClass = 3 OR QuadClass = 4)
        AND DATEADDED >= (SELECT MAX(DATEADDED) - 1000000 FROM `gdelt-bq.gdeltv2.events`)
    ORDER BY 
        DATEADDED DESC
    """

    query_job = client.query(sql_query).result()

    data = [{
    'GlobalEventID': row.GLOBALEVENTID,
    'Actor1Geo_Fullname': row.Actor1Geo_FullName,
    'Actor2Geo_Fullname': row.Actor2Geo_FullName,
    'ActionGeo_Fullname': row.ActionGeo_FullName,
    'NumSources': row.NumSources,
    'NumMentions' : row.NumMentions,
    'NumArticles': row.NumArticles,
    'QuadClass': row.QuadClass,
    'GoldsteinScale': row.GoldsteinScale,
    'AvgTone': row.AvgTone,
    'DateAdded': row.DATEADDED
    } for row in query_job]

    print("Query Complete in --- %s seconds ---" % (time.time() - start_time))
    return data


def fetch_new_data():
    # Construct SQL query to fetch new data added in the last 15 minutes
    sql_query = """
    SELECT 
        GLOBALEVENTID, Actor1Geo_FullName, Actor2Geo_FullName, ActionGeo_FullName,
        NumSources, NumMentions, NumArticles, QuadClass, GoldsteinScale, AvgTone, DATEADDED
    FROM 
        `gdelt-bq.gdeltv2.events`
    WHERE
        Actor1Geo_FullName IS NOT NULL 
        AND Actor2Geo_FullName IS NOT NULL 
        AND ActionGeo_FullName IS NOT NULL
        AND (QuadClass = 3 OR QuadClass = 4)
        AND DATEADDED >= (SELECT MAX(DATEADDED) - 0001500 FROM `gdelt-bq.gdeltv2.events`)
    ORDER BY 
        DATEADDED DESC
    """
    # Execute the query
    query_job = client.query(sql_query).result()

    data = [{
    'GlobalEventID': row.GLOBALEVENTID,
    'Actor1Geo_Fullname': row.Actor1Geo_FullName,
    'Actor2Geo_Fullname': row.Actor2Geo_FullName,
    'ActionGeo_Fullname': row.ActionGeo_FullName,
    'NumSources': row.NumSources,
    'NumMentions' : row.NumMentions,
    'NumArticles': row.NumArticles,
    'QuadClass': row.QuadClass,
    'GoldsteinScale': row.GoldsteinScale,
    'AvgTone': row.AvgTone,
    'DateAdded': row.DATEADDED
    } for row in query_job]

    print("Query Complete in --- %s seconds ---" % (time.time() - start_time))
    return data

def Country(location):
    comma_count = location.count(',')
    parts = location.split(',')
    
    if comma_count == 0:
        return location.strip()
    elif comma_count == 1:
        return parts[1].strip()
    else:  # comma_count is 2 or more
        return parts[2].strip()


def preprocess_data(data):
    # Apply feature engineering on the fetched data
    # Your preprocessing code here
    df = pd.DataFrame(data)

    events_updated = df.copy()
    events_updated['Actor1Name'] = events_updated['Actor1Geo_Fullname'].apply(lambda x: Country(x) if pd.notnull(x) else x) 
    events_updated['Actor2Name'] = events_updated['Actor2Geo_Fullname'].apply(lambda x: Country(x) if pd.notnull(x) else x)
    events_updated['ActionName'] = events_updated['ActionGeo_Fullname'].apply(lambda x: Country(x) if pd.notnull(x) else x) 

    # Correct the column names in the drop method
    events_updated.drop(columns=['Actor1Geo_Fullname', 'Actor2Geo_Fullname', 'ActionGeo_Fullname'], inplace=True)  # Corrected 'Actor2Geo_Fullname'

    events_updated['ConflictType'] = '1'
    events_updated.loc[events_updated['Actor1Name'] == events_updated['Actor2Name'], 'ConflictType'] = '0'

    events_updated['Reach'] = (events_updated['NumMentions'] +events_updated['NumSources'])/2 
    events_updated['impact_verifier'] = (0.6* events_updated['AvgTone']) + (0.4* events_updated['GoldsteinScale'])
    threshold = events_updated['impact_verifier'].quantile(0.40)
    events_updated['ConflictClass'] = (events_updated['impact_verifier'] >= threshold).astype(int)  # 1 for major conflicts, 0 for minor

    events_updated['Actor1Name'] = events_updated['Actor1Name'].astype(str)
    events_updated['Actor2Name'] = events_updated['Actor2Name'].astype(str)
    events_updated['ActionName'] = events_updated['ActionName'].astype(str)

    print("Building DF in --- %s seconds ---" % (time.time() - query_time))
    return events_updated

#frequency encoding for the reamining categorical variable
def frequency_encoding(data, feature):
  """
  Encodes a categorical feature using frequency encoding.

  Args:
      data (pandas.DataFrame): The DataFrame containing the feature to encode.
      feature (str): The name of the categorical feature to encode.

  Returns:
      pandas.DataFrame: The DataFrame with the encoded feature.
  """

  def encode_row(row):
    value = row[feature]
    count = data[feature].value_counts().get(value, 0)  # Handle missing values
    return count / len(data)

  # Apply the encoding function row-wise
  data[feature] = data.apply(encode_row, axis=1)

  return data

def train_model(data):
    numerical_cols = ['GoldsteinScale','AvgTone','Reach','NumMentions','NumSources','NumArticles']
    actor1 = data['Actor1Name']
    actor2 = data['Actor2Name']
    actions = data['ActionName']
    dates = data['DateAdded']
    data.drop(columns='DateAdded', inplace=True)

    scaler = StandardScaler()
    scaledNumericalColumns = scaler.fit_transform(data[numerical_cols])
    data[numerical_cols] = scaledNumericalColumns

    le = LabelEncoder()
    data['ConflictType'] = le.fit_transform(data['ConflictType'])

    print("Starting")
    data = frequency_encoding(data.copy(), 'Actor1Name')
    print("done 1")
    data = frequency_encoding(data.copy(), 'Actor2Name')
    print("done 2")
    data = frequency_encoding(data.copy(), 'ActionName')
    print("done 3")

    X = data.drop('ConflictClass', axis=1) # features
    X2 = data.drop('ConflictClass', axis=1) # features

    #Perform clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X2)

    # Evaluate clustering using silhouette score
    silhouette_avg = silhouette_score(X2, data['Cluster'])
    print("Silhouette Score:", silhouette_avg)
    ch_score = calinski_harabasz_score(X2, data['Cluster'])
    db_score = davies_bouldin_score(X2, data['Cluster'])
    silhouette_avg = silhouette_score(X2, data['Cluster'])
    ari_score = adjusted_rand_score(data['ConflictClass'], data['Cluster'])
    ami_score = adjusted_mutual_info_score(data['ConflictClass'], data['Cluster'])

    print("Calinski-Harabasz Score K-means:", ch_score)
    print("Davies-Bouldin Score K-means:", db_score)
    print("Silhouette Score K-means:", silhouette_avg)
    print("Adjusted Rand Index K-means:", ari_score)
    print("Adjusted Mutual Information K-means:", ami_score)
    
    print("Encode & Train Model in --- %s seconds ---" % (time.time() - build_t))

    sentiments = kmeans.predict(X)
    data['Sentiment'] = sentiments
    data.loc[data['QuadClass'] == 3, 'QuadClass'] = 'Verbal Conflict'
    data.loc[data['QuadClass'] == 4, 'QuadClass'] = 'Material Conflict'

    data.loc[(data['Sentiment'] == 0) & (data['ConflictType'] == 0), 'Sentiment'] = 'Minor Rise in Civil Affairs'
    data.loc[(data['Sentiment'] == 1) & (data['ConflictType'] == 0), 'Sentiment'] = 'Major Rise in Civil Affairs'
    data.loc[(data['Sentiment'] == 0) & (data['ConflictType'] == 1), 'Sentiment'] = 'Minor Friction Rising Between Nations'
    data.loc[(data['Sentiment'] == 1) & (data['ConflictType'] == 1), 'Sentiment'] = 'Major Friction Rising Between Nations'
    data.loc[data['ConflictType'] == 0, 'ConflictType'] = 'Internal Conflict'
    data.loc[data['ConflictType'] == 1, 'ConflictType'] = 'External Conflict'
    data['Actor1Name'] = actor1
    data['Actor2Name'] = actor2
    data['ActionName'] = actions
    data['DateAdded'] = pd.to_datetime(dates, format='%Y%m%d%H%M%S')

    pandas_gbq.to_gbq(data, table_name, project_id=project_id, if_exists='replace')

    return kmeans



def classify_data(model, data):
    # Use the trained model to classify the preprocessed data
    # Your classification code here
    numerical_cols = ['GoldsteinScale','AvgTone','Reach','NumMentions','NumSources','NumArticles']
    dates = data['DateAdded']
    data.drop(columns='DateAdded', inplace=True)

    scaler = StandardScaler()
    scaledNumericalColumns = scaler.fit_transform(data[numerical_cols])
    data[numerical_cols] = scaledNumericalColumns

    le = LabelEncoder()
    data['ConflictType'] = le.fit_transform(data['ConflictType'])

    actor1 = data['Actor1Name']
    actor2 = data['Actor2Name']
    actions = data['ActionName']

    print("Starting")
    data = frequency_encoding(data.copy(), 'Actor1Name')
    print("done 1")
    data = frequency_encoding(data.copy(), 'Actor2Name')
    print("done 2")
    data = frequency_encoding(data.copy(), 'ActionName')
    print("done 3")

    X = data.drop('ConflictClass', axis=1) # features

    sentiments = model.predict(X)
    data['Sentiment'] = sentiments
    data.loc[data['QuadClass'] == 3, 'QuadClass'] = 'Verbal Conflict'
    data.loc[data['QuadClass'] == 4, 'QuadClass'] = 'Material Conflict'

    data.loc[(data['Sentiment'] == 0) & (data['ConflictType'] == 0), 'Sentiment'] = 'Minor Rise in Civil Affairs'
    data.loc[(data['Sentiment'] == 1) & (data['ConflictType'] == 0), 'Sentiment'] = 'Major Rise in Civil Affairs'
    data.loc[(data['Sentiment'] == 0) & (data['ConflictType'] == 1), 'Sentiment'] = 'Minor Friction Rising Between Nations'
    data.loc[(data['Sentiment'] == 1) & (data['ConflictType'] == 1), 'Sentiment'] = 'Major Friction Rising Between Nations'
    data.loc[data['ConflictType'] == 0, 'ConflictType'] = 'Internal Conflict'
    data.loc[data['ConflictType'] == 1, 'ConflictType'] = 'External Conflict'
    data['Actor1Name'] = actor1
    data['Actor2Name'] = actor2
    data['ActionName'] = actions
    data['DateAdded'] = pd.to_datetime(dates, format='%Y%m%d%H%M%S')

    return data

def write_predictions_to_bigquery(data_w_sentiment):
    # Write the predictions back to your BigQuery table
    # Your code to write predictions to BigQuery
    # Check if the GlobalEventID already exists in the BigQuery table
    existing_ids = list(data_w_sentiment['GlobalEventID'])
    existing_ids_str = ','.join([str(event_id) for event_id in existing_ids])

    sql_query = f"""
    SELECT GlobalEventID
    FROM `dp-project-419414.Sentiment_dataset.Sentiment_table`
    WHERE GlobalEventID IN ({existing_ids_str})
    """
    existing_ids_in_table = pd.read_gbq(sql_query, project_id=project_id)['GlobalEventID'].tolist()

    # Update existing rows if GlobalEventID exists, otherwise append new rows
    if len(existing_ids_in_table) != 0:
        print("Found Dup")
        # Update existing rows
        for id in existing_ids_in_table:
            event_data = data_w_sentiment[data_w_sentiment['GlobalEventID'] == id].iloc[0]  # Get data for the event
            update_query = f"""
            UPDATE `dp-project-419414.Sentiment_dataset.Sentiment_table`
            SET NumSources = {event_data['NumSources']},
                NumMentions = {event_data['NumMentions']},
                NumArticles = {event_data['NumArticles']},
                QuadClass = {event_data['QuadClass']},
                GoldsteinScale = {event_data['GoldsteinScale']},
                AvgTone = {event_data['AvgTone']},
                Actor1Name = {event_data['Actor1Name']},
                Actor2Name = {event_data['Actor2Name']},
                ActionName = {event_data['ActionName']},
                ConflictType = '{event_data['ConflictType']}',
                Reach = {event_data['Reach']},
                ConflictClass = {event_data['ConflictClass']},
                Sentiment = '{event_data['Sentiment']}'
            WHERE GlobalEventID = '{id}'
            """
            # Execute the update query
            client.query(update_query)
            
        new_rows = data_w_sentiment[~data_w_sentiment['GlobalEventID'].isin(existing_ids_in_table)]
        pandas_gbq.to_gbq(new_rows, table_name, project_id=project_id, if_exists='append')
    else:
        # Append new rows
        pandas_gbq.to_gbq(data_w_sentiment, table_name, project_id=project_id, if_exists='append')





# Initialize BigQuery client
start_time = time.time()
credentials_path = 'dp-project-419414-ee4c9f823c6e.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

client = bigquery.Client()
project_id = 'dp-project-419414'
table_name = 'dp-project-419414.Sentiment_dataset.Sentiment_table'

init_data = initialize_ds()

query_time = time.time()
preprocessed_data = preprocess_data(init_data)

build_t = time.time()
model = train_model(preprocessed_data.copy())

time.sleep(900)

while True:
    # Fetch new data
    f_u_time = time.time()
    new_data = fetch_new_data()

    if new_data:
        # Preprocess the fetched data
        query_time = time.time()
        preprocessed_data = preprocess_data(new_data)

        # Classify the preprocessed data using the trained model
        build_t = time.time()
        predictions = classify_data(model, preprocessed_data)

        # Write predictions to BigQuery
        write_predictions_to_bigquery(predictions)
        print("Fetched and uploaded new data in --- %s seconds ---" % (time.time() - f_u_time))

    # Sleep for 15 minutes before fetching new data again
    time.sleep(900)  # 900 seconds = 15 minutes

