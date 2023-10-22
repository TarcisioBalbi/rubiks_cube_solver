from utils.data_generation import dataGenerator

cubeDF = dataGenerator(n_runs=1000,n_moves=15,random_state=10)
cubeDF.to_parquet('data/cubeDS.parquet')
