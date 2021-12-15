from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import concat, lit
from datetime import datetime, timedelta
import json
import numpy as np
import sys

def main(sc, spark):
    dfPlaces = spark.read.csv('/data/share/bdm/core-places-nyc.csv', header=True, escape='"')
    dfPattern = spark.read.csv('/data/share/bdm/weekly-patterns-nyc-2019-2020/*', header=True, escape='"')
    OUTPUT_PREFIX = sys.argv[1]
    
    CAT_CODES = {'445210', '722515', '445299', '445120', '452210', '311811', '722410', '722511', '445220', 
                '445292', '445110', '445291', '445230', '446191', '446110', '722513', '452311'}
    CAT_GROUP = {'452311': 0, '452210': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446191': 5,
                '446110': 5, '722515': 6, '311811': 6, '445299': 7, '445220': 7, '445292': 7, '445291': 7, '445230': 7, '445210': 7, '445110': 8}

    dfD = dfPlaces.filter(dfPlaces.naics_code.isin(CAT_CODES)).select('placekey', 'naics_code')
    
    udfToGroup = F.udf(lambda x: CAT_GROUP[x])

    dfE = dfD.withColumn('group', udfToGroup('naics_code'))
    
    dfF = dfE.drop('naics_code').cache()
    
    def expandVisits(date_range_start, visits_by_day):
      visits_by_day = visits_by_day.replace('[', '').replace(']', '').split(',')
      for i in range(7):
        datem = datetime.strptime(date_range_start[0:10], "%Y-%m-%d")  + timedelta(days=i)
        date = datem.strftime("%m-%d")
        yield(datem.year, date, int(visits_by_day[i]))

    visitType = T.StructType([T.StructField('year', T.IntegerType()),
                              T.StructField('date', T.StringType()),
                              T.StructField('visits', T.IntegerType())])

    udfExpand = F.udf(expandVisits, T.ArrayType(visitType))

    dfH = dfPattern.join(dfF, 'placekey') \
        .withColumn('expanded', F.explode(udfExpand('date_range_start', 'visits_by_day'))) \
        .select('group', 'expanded.*')

    def computeStats(group, visits):
        median_val = int(median(visits))
        high_val = max(visits)
        low_val = min(visits)

        return(median_val, low_val, high_val)

    statsType = T.StructType([T.StructField('median', T.IntegerType()),
                              T.StructField('low', T.IntegerType()),
                              T.StructField('high', T.IntegerType())])

    udfComputeStats = F.udf(computeStats, statsType)

    dfI = dfH.groupBy('group', 'year', 'date') \
        .agg(F.collect_list('visits').alias('visits')) \
        .withColumn('stats', udfComputeStats('group', 'visits'))

    dfJ = dfI \
        .withColumn('median', dfI.select('stats')[0]['median']) \
        .withColumn('low', dfI.select('stats')[0]['low']) \
        .withColumn('high', dfI.select('stats')[0]['high']) \
        .withColumn('date', concat(dfI.year, lit('-'), dfI.date)) \
        .drop('visits', 'stats') \
        .cache()
        

    filenames = ['big_box_grocers', 'convenience_stores', 'drinking_places', 'full_service_resturants', 'limited_service_resturants', 
                 'pharmacies_and_drug_stores', 'snack_and_bakeries', 'specialty_food_stores', 'supermarkets_except_convenience_stores']

    for i in range(9):
      dfJ.filter(f'group={i}') \
        .drop('group') \
        .coalesce(1) \
        .write.csv(f'{OUTPUT_PREFIX}/{filenames[i]}',
                   mode='overwrite', header=True)

if __name__=='__main__':
    sc = SparkContext()
    spark = SparkSession(sc)
    main(sc, spark)