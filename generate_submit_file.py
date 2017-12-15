import os
import optparse as op

if __name__=="__main__":
    parser = op.OptionParser()
    parser.add_option("-f", "--folder", type="string", dest="folder", help="folder that contains inputs and will contain outputs")
    parser.add_option("-N", type="int", dest="ranks", help="number of galaxies to rank")
    parser.add_option("--dmax", type="float", dest="dmax", help="maximum distance (Mpc)")
    parser.add_option("--max-stick", type="int", dest="max_stick", help="maximum number of gaussian components")
    parser.add_option("-n", type="int", dest="nevents", help="number of events to analyse")
    parser.add_option("-o", type="string", dest="condor_name", help="name of the condor submit file")
    parser.add_option("-l", type="string", dest="logdir", help="log dir")
    parser.add_option("--bin",type="string", dest="PATH", help="location of the executable")
    (options, args) = parser.parse_args()

    string="""
        universe = vanilla
        executable = %s/VolumeLocalisation.py
        output = %s/out.$(Process)
        error = %s/err.$(Process)
        log = %s/log.$(Process)
        notification = Never
        getenv = True
        request_memory = 20000
        request_cpus = 8
        """%(options.PATH,options.logdir,options.logdir,options.logdir)


    for i in range(1,options.nevents+1):
        # find the input file
        posterior_folder = os.path.join(options.folder,str(i),'post')
        allfiles = os.listdir(posterior_folder)
        for file in allfiles:
            if 'posterior' in file and 'B' not in file: posterior_file = file
        input_file = os.path.join(posterior_folder,posterior_file)
        # find the injection file
        injections_file = os.path.join(options.folder,str(i),str(i)+'.xml')
        # setup the output folder
        output_file = os.path.join(options.folder,str(i),'skypos')
    
        string +="arguments = --catalog GLADE_1.0.txt -i %s -o %s --inj %s --bins 100,360,720 -e %s --max-stick %d --dmax %f -N %d\n"%(input_file,output_file,injections_file,i,options.max_stick,options.dmax,options.ranks)
        string+="accounting_group = ligo.dev.o1.cbc.pe.lalinference\n"
        string+="queue 1\n"

    f=open(options.condor_name,"w")
    f.write(string)
    f.close()
