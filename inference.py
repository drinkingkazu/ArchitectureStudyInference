import matplotlib 
matplotlib.use('Agg')

import os,commands,signal,sys,tempfile
if not 'HBD_CHURCH_DIR' in os.environ:
    sys.stderr.write('Cannot use inference.py w/o HBD_CHURCH_DIR set...\n')
    sys.exit(1)
os.environ['GLOG_minloglevel'] = '2' # set message level to warning
import pandas
import caffe
import numpy as np
import ROOT as rt
import lmdb
import time
from ROOT import larcv
import matplotlib.pyplot as plt
caffe.set_mode_gpu()
### global variable: csv data field definition
CSV_VARS = ['entry',
            'energy_dep','energy_start',
            'mass','mom_start',
            'dcosx_start','dcosy_start','dcosz_start',
            'nparticle',
            'ndecay',
            'npx',
            'label','prediction',
            'eminus','gamma','muminus','piminus','proton']

### caffe inference class for SPClassification
class inference:

    ### class attributes to be overriden by instance
    _proto, _weight, _batch_size, _filler_name, _mem_limit, _output = [None]*6
    ### class attribute not to be overriden by instance
    _terminate = False

    ### constructor sets (invalid) default values, override class attributes w/ instance attributes
    def __init__(self):
        self._proto = self._weight = self._batch_size = self._filler_name = self._output = None
        self._mem_limit = 10000

    ### 3 parameters from a user to configure: prototxt for network def, weight for inference, and output csv file name
    def configure(self,proto,weight,output,io_cfg):
        try:
            self._proto  = str(proto)
            self._weight = str(weight)
            self._output = str(output)
        except TypeError:
            print '\033[93mWrong type in configure argument\033[00m:',proto,weight,output
            raise TypeError
        # check if proto exists
        if not os.path.isfile(self._proto):
            print '\033[93mPROTO does not exist\033[00m:',self._proto
            self._proto = None
            return
        # check if weight exists
        if not os.path.isfile(self._weight):
            print '\033[93mWEIGHT does not exist\033[00m:',self._weight
            self._proto  = None
            self._weight = None
            return
        # check if IO config exists
        if not os.path.isfile(io_cfg):
            print '\033[93mFILLER CFG does not exist\033[00m:',io_cfg
            self._proto  = None
            self._weight = None
            return
        # check if output exists
        if os.path.isfile(self._output):
            print '\033[95mWarning ... output already exists\033[00m:',output
            try:
                df = pandas.read_csv(self._output)
                # check if data definition makes sense
                if not len(df.axes[1]) == len(CSV_VARS):
                    raise Exception
                for v in CSV_VARS: exec('df.%s.values' % v)
                print 'Will skip existing entries...'
            except Exception:
                print '\033[95mOutput is an invalid csv file. Will overwrite...\033[00m'
                os.remove(self._output)

        # parse proto to get som info since I'm as dirty as fuck
        # filler is a string key to get a handle on caffe's using LArCV io processor
        
        tempf = tempfile.NamedTemporaryFile(delete=False)
        for l in open(self._proto,'r').read().split('\n'):
            if len(l.split())<1: continue
            if l.split()[0].startswith('#'): continue

            if l.find('batch_size')>=0:
                try:
                    self._batch_size = int(l.replace(':',' ').split()[1])
                except Exception:
                    print '\033[93mFailed to infer batch_size\033[00m:',l
            if l.find('filler_name')>=0:
                try:
                    self._filler_name = str(l.replace(':',' '))
                    self._filler_name = self._filler_name.replace('"','')
                    self._filler_name = self._filler_name.split()[1]
                except Exception:
                    print '\033[93mFailed to infer filler_name\033[00m:',l
            outl = str(l)
            if l.find('filler_config')>=0:
                outl = 'filler_config: "%s"' % io_cfg
            tempf.write('%s\n' % outl)
        self._proto = tempf.name

    ### single function to check the state to see if run() is callable, not for users to understand
    def _ready_(self):
        if   not self._proto:       print '\033[93mPROTO is unspecified (configure function not called or failed)\033[00m'
        elif not self._weight:      print '\033[93mWEIGHT is unspecified (configure function not called or failed\033[00m'
        elif not self._weight:      print '\033[93mFILLER CFG is unspecified (configure function not called or failed\033[00m'
        elif not self._batch_size:  print '\033[93mbatch_size could not be found in PROTO (failed parsing in configure function)\033[00m'
        elif not self._filler_name: print '\033[93mfiller_name could not be found in PROTO (failed parsing in configure function)\033[00m'
        else: return True
        return False

    ### taken from pubs script, fetch gpu info, not for users to understand
    def _ls_gpu_(self):
        output = commands.getoutput('nvidia-smi | grep MiB')
        mem_usage = {}
        mem_max   = []
        for l in output.split('\n'):
            words=l.split()
            if len(words) < 4: continue

            if words[1].isdigit():
                gpu_id  = int(words[1])
                if not gpu_id in mem_usage:
                    mem_usage[gpu_id] = 0
                mem_usage[gpu_id] += int(words[-2].replace('MiB',''))
            else:
                mem_max.append(int(words[-5].replace('MiB','')))

            for i in xrange(len(mem_max)):
                if not i in mem_usage:
                    mem_usage[i] = 0

            for i in mem_usage:
                assert i < len(mem_max)

        return (mem_usage,mem_max)

    ### taken from pubs script, pick an available gpu, not for users to understand
    def _pick_gpu_(self):
        gpu_info = self._ls_gpu_()
        for gpu in gpu_info[0]:
            mem_available = gpu_info[1][gpu] - gpu_info[0][gpu]
            if mem_available > self._mem_limit: 
                return (len(gpu_info[0])-1) - gpu
        return -1

    ### main run function to be called after configure()
    def run(self):

        # check if configuration is set
        if not self._ready_():
            print '\033[93mAborting\033[00m'
            return

        # check which gpu to use
        gpu = self._pick_gpu_()
        if gpu<0:
            print '\033[93mNo GPU available...\033[93m'
            return
        caffe.set_device(gpu)

        # load a list of pre-processed events, if output already exists
        # also prepare output fstream descriptor pointer
        done_list=None
        fout=None
        if os.path.isfile(self._output):
            df = pandas.read_csv(self._output)
            done_list = [int(x) for x in df.entry.values.astype(np.uint32)]
            fout=open(self._output,'a')
        else:
            fout=open(self._output,'w')
            line=''
            for v in CSV_VARS:
                line += '%s,' % v
            fout.write(line.rstrip(',') + '\n')
            
        # construct a net, this also configures internal larcv IO processor
        net = caffe.Net( self._proto, self._weight, caffe.TEST)
        
        # check if larcv IO processor does in fact exist and registered in a factory
        if not larcv.ThreadFillerFactory.exist_filler(self._filler_name):
            print '\033[93mFiller',self._filler_name,'does not exist...\033[00m'
            return
            
        # get IO instance, ThreadDatumFiller instance, from the factory
        filler = larcv.ThreadFillerFactory.get_filler(self._filler_name)

        # get # events to be processed 
        num_events = filler.get_n_entries()
        
        # force random access to be false for an inference
        filler.set_random_access(False)

        # construct our own IO to fetch ROI object for physics analysis, use RED mode w/ same input files
        myio = larcv.IOManager(0,"AnaIO")
        for f in filler.pd().io().file_list():
            myio.add_in_file(f)
        myio.initialize()

        print
        print '\033[95mTotal number of events\033[00m:',num_events
        print '\033[95mBatch size\033[00m:', self._batch_size
        print

        event_counter = 0    # this variable denotes which TTree entry we are @ in the loop below
        stop_counter  = 1e10 # well, unused, but one can set a break condition by configuring this parameter

        # now continue a loop till the end of the input file (event list)
        while 1:

            # if previous result is loaded, check if we should process the current entry or not
            if done_list and (event_counter in done_list):
                event_counter+=1
                continue

            # force the filler to move the next event-to-read pointer to the entry of our interest
            filler.set_next_index(event_counter)

            # number of entries we expect to process in this mini-batch
            num_entries = num_events - event_counter
            if num_entries > self._batch_size: 
                num_entries = self._batch_size

            # now run the network for a mini-batch, sleep while the thread is running
            net.forward()
            while filler.thread_running():
                time.sleep(0.001)

            # retrieve ROI product producer from the filler, so we can read-in ROI products through myroi 
            roi_producer = filler.producer(1)

            # get a vector of integers that record TTree entry numbers processed in this mini-batch
            entries = filler.processed_entries()
            if entries.size() != self._batch_size:
                print "\033[93mBatch counter mis-match!\033[00m"
                raise Exception

            # retrieve data already read-and-stored-in-memory from caffe blob
            adcimgs = net.blobs["data"].data    # this is image
            labels  = net.blobs["label"].data   # this is label
            scores  = net.blobs["softmax"].data # this is final output softmax vector
            
            # loop over entry of mini-batch outcome
            for index in xrange(num_entries):
                
                if not entries[index] == event_counter:
                    print '\033[93mLogic error... inconsistency found in expected entry (%d) vs. processing entry (%d)' % (event_counter,entries[index])
                    self.__class__._terminate = True
                    break
                # skip if this is alredy recorded entry
                if done_list and (event_counter in done_list):
                    event_counter +=1
                    continue

                # update an user which entry we are processing
                sys.stdout.write('Processing entry %d\r' % event_counter)
                
                # declare csv_vals dictionary instance, and fill necessary key-value pairs.
                # later we have an explicit check if all keys are filled.
                # this is helpful to avoid a mistake when someone udpate later the script
                # to include/exclude variables in CSV_VARS definition and forgot to update this
                # portion of the code.
                csv_vals={}
                adcimg = adcimgs[index] # ADC raw image
                label  = labels[index]  # Labels
                score  = scores[index]  # results
                # fill things that can be filled from caffe blob
                csv_vals['entry'  ] = entries[index]
                csv_vals['npx'    ] = (adcimg > 0).sum()
                csv_vals['label'  ] = int(label)
                csv_vals['prediction'] = score.argmax()
                csv_vals['eminus' ] = score[0]
                csv_vals['gamma'  ] = score[1]
                csv_vals['muminus'] = score[2]
                csv_vals['piminus'] = score[3]
                csv_vals['proton' ] = score[4]
                
                # now get ROI data from myroi, our separate IO handle, to record physics parameters
                myio.read_entry(entries[index])
                event_roi = myio.get_data(1,roi_producer)
                
                csv_vals['nparticle']=0
                csv_vals['ndecay']=0
                csv_vals['energy_dep']=0.
                # loop over ROIs
                for roi in event_roi.ROIArray():
                    if roi.MCSTIndex() == larcv.kINVALID_USHORT:
                        # ROI from simb::MCTruth
                        csv_vals['energy_start']=roi.EnergyInit()
                        csv_vals['mass'] = larcv.ParticleMass(roi.PdgCode())
                        px,py,pz = (roi.Px(),roi.Py(),roi.Pz())
                        ptot = np.sqrt(np.power(px,2)+np.power(py,2)+np.power(pz,2))
                        csv_vals['mom_start'] = ptot
                        csv_vals['dcosx_start'] = px/ptot
                        csv_vals['dcosy_start'] = py/ptot
                        csv_vals['dcosz_start'] = pz/ptot
                    else:
                        # ROI from sim::MCShower and sim::MCTrack
                        csv_vals['nparticle']+=1
                        if roi.ParentTrackID() == roi.TrackID():
                            csv_vals['energy_dep'] = roi.EnergyDeposit()
                        elif np.abs(roi.PdgCode()) == 13 and np.abs(roi.ParentPdgCode()) == 211:
                            csv_vals['ndecay'] += 1
                        elif np.abs(roi.PdgCode()) == 11 and np.abs(roi.ParentPdgCode()) == 13:
                            csv_vals['ndecay'] += 1
                # record in csv format
                line = ''
                for v in CSV_VARS:
                    try:
                        line += '%s,' % str(csv_vals[v])
                    except KeyError:
                        print '\033[93mCould not locate field\033[00m:',v
                        self.__class__._terminate=True
                        break
                line=line.rstrip(',')
                line+='\n'
                fout.write(line)

                # break if stop counter is met
                event_counter += 1
                if event_counter >= stop_counter:
                    break
                # break if termination is called
                if self.__class__._terminate:
                    break

            # break if all entries are processed
            if num_entries < self._batch_size:
                break
            # break if stop counter is met
            if event_counter >= stop_counter:
                break
            # break if termination is called
            if self.__class__._terminate:
                print
                print '\033[93mAborting upon kernel kill signal...\033[00m'
                break
        print
        # close outputs and input io
        fout.close()
        myio.finalize()
        # destroy thread filler via factory, an owner
        larcv.ThreadFillerFactory.destroy_filler(self._filler_name)

### register ctrl+C handling cuz I'm lazy and often kill stuffs
def sig_kill(signal,frame):
    inference._terminate = True
    print '\033[95mSIGINT detected.\033[00m Finishing the program gracefully.'
    print 'Terminating proc_daemon::routine function.'

signal.signal(signal.SIGINT,  sig_kill)
signal.signal(signal.SIGQUIT, sig_kill)
signal.signal(signal.SIGTERM, sig_kill)

# re-implement the driver component as you wish as fuck. here, just a dumb simple main method
if __name__ == '__main__':

    p = inference()
    WEIGHT,PROTO,IO_CFG = (None,None,None)
    for a in sys.argv:
        if a.find('caffemodel') >=0:
            WEIGHT=str(a)
        if a.find('prototxt') >=0:
            PROTO=str(a)
        if a.find('cfg') >=0:
            IO_CFG=str(a)

    # output directory
    outdir = IO_CFG[IO_CFG.rfind('/')+1:IO_CFG.find('.cfg')]
    os.system('mkdir -p %s' % outdir)
    if not os.path.isdir(outdir):
        print 'Failed to make an output directory:',outdir
    # make up an output file name in mixture of weight file and filler config
    output = outdir + '/' + WEIGHT[WEIGHT.rfind('/')+1:WEIGHT.find('caffemodel')] + 'csv'
    p.configure(proto=PROTO,weight=WEIGHT,output=output,io_cfg=IO_CFG)
    p.run()

