#!/bin/bash

python eval.py algo_config='td3+bc' +model_path=model_td3+bc_random_001_0.d3
python eval.py algo_config='td3+bc' +model_path=model_td3+bc_random_001_1.d3
python eval.py algo_config='td3+bc' +model_path=model_td3+bc_random_001_2.d3

python eval.py algo_config="dt" ++model_path=model_dt_random_001_0.d3
python eval.py algo_config="dt" ++model_path=model_dt_random_001_1.d3
python eval.py algo_config="dt" ++model_path=model_dt_random_001_2.d3

python eval.py algo_config="cql" ++model_path=model_cql_random_001_0.d3
python eval.py algo_config="cql" ++model_path=model_cql_random_001_1.d3
python eval.py algo_config="cql" ++model_path=model_cql_random_001_2.d3

