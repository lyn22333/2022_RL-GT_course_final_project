# 2022_RL-GT_course_final_project
This is the repository of my final project in 2022 RL&GT course.


I implemented MAPPO, CPPO, IPPO on VMAS benchmark(https://github.com/proroklab/VectorizedMultiAgentSimulator). 
I referred to the official MAPPO source code(https://github.com/marlbenchmark/on-policy), 
and tried to make some improvement using Beta distribution.

## How to start
* install vmas

  ```pip install vmas```

* run the corresponding file

  for example:

  ```python run_ippo.py```

  You can change the parameters in run_ippo.py, run_mappo.py and run_cppo.py.
Remember to create correspongding dir in './log' if you change the `log_file`
