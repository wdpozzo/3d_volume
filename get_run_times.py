import numpy as np
import os
import datetime

if __name__=="__main__":
    indir = 'log_2015'
    all_files = os.listdir(indir)
    log_files = [f for f in all_files if 'log' in f]
    run_times = []
    for l in log_files:
        f = open(os.path.join(indir,l),'r')
	for line in f:
	    if 'executing' in line:
                tmp = line.split(None)[3]
		tmp2 = np.array(tmp.split(':'),dtype=np.double)
		start_time = tmp2[0]*60.0*60.0+tmp2[1]*60.0+tmp2[2]
	    if 'terminated' in line:
                tmp = line.split(None)[3]
                tmp2 = np.array(tmp.split(':'),dtype=np.double)
                end_time = tmp2[0]*60.0*60.0+tmp2[1]*60.0+tmp2[2]
	f.close()
   	run_times.append(end_time-start_time)
    print indir,": mean:",np.mean(run_times),"std:", np.std(run_times),"percentiles [5,50,95]:", np.percentile(run_times,[5.0,50.0,95.0])

