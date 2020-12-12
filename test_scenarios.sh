#!/bin/bash

# RUN SIMULATIONS FOR SVD BASED METHOD
for i in {1..12}
do 
    python test_scenarios.py --algorithm svd --samples 4000 --test_case case$i --csv_file $1
done

# RUN SIMULATIONS FOR Q_METHOD METHOD
for i in {1..12}
do 
    python test_scenarios.py --algorithm q_method --samples 4000 --test_case case$i --csv_file $1
done

# RUN SIMULATIONS FOR QUEST METHOD
for i in {1..12}
do 
    python test_scenarios.py --algorithm quest --iterations 0 --samples 4000 --test_case case$i --csv_file $1
done

# RUN SIMULATIONS FOR ESOQ2 METHOD
for i in {1..12}
do 
    python test_scenarios.py --algorithm esoq2 --iterations 0 --samples 4000 --test_case case$i --csv_file $1
done