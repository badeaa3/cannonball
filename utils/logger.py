'''
General logger class for monitoring and storing the training progress.
'''

from numpy import mean 
import pickle
import torch
import numpy as np
from time import time

class Logger():

    def __init__(self,
                 logs                           = None,
                 warnings                       = None,
                 errors                         = None, 
                 hooks                          = None, 
                 model                          = None,
                 nepoch                         = None,
                 niter                          = None,
                 earlystop_epsilon              = None,
                 earlystop_lookback             = None, # number of epochs to look back to use above epsilon
                 earlystop_within_epoch_epsilon = None,
                 earlystop_patience             = None, # number of iterations with an epoch to wait before using above epsilon
                 eval_mode                      = False
                 ):
        self.logs     = logs
        self.warnings = warnings
        self.errors   = errors
        self.hooks    = hooks
        if hooks and model: 
            self.register_hooks(model)
        self.history = {
            "params"      : {},
            "grads"       : {}, 
            "activations" : {},
            "timestamps"  : [[],[]],
            "train"       : self.stat_dict(), 
            "val"         : self.stat_dict()
        }
        self.recording  = True
        self.nepoch     = nepoch
        self.niter      = niter
        self.ntotal     = float(nepoch*niter)
        self.starttime  = 0 

        # early stop settings
        self.earlystop_epsilon              = earlystop_epsilon
        self.earlystop_lookback             = earlystop_lookback
        self.earlystop_within_epoch_epsilon = earlystop_within_epoch_epsilon
        self.earlystop_patience             = earlystop_patience

        # evaluation mode don't print stats
        self.eval_mode   = eval_mode
    def stat_dict(self):
        return {"loss_isr"          : [],
                "loss_decay"        : [],
                "loss_total"        : [],
                "acc_max_isr"       : [],
                "acc_max_decay"     : [],
                "acc_max_total"     : [],
                "acc_thresh_isr"    : [],
                "acc_thresh_decay"  : [],
                "acc_thresh_total"  : []}

    def record_activations(self, activations, module_name):
        '''forward hook for network sub-modules'''
        if not self.recording: return
        self.history["activations"][module_name].append(activations.clone())
            
    def record_params(self, module, param_name):
        '''forward hook for network super-module'''
        if not self.recording: return
        param = module.state_dict()[param_name]
        self.history["params"][param_name].append(param.clone())
                                
    def record_grads(self, grad, param_name):
        '''backward hook for Tensors with gradient'''
        if not self.recording: return
        if grad == None: return
        self.history["grads"][param_name].append(grad.clone())
            
    def save_history(self, filepath="./logs/history.pickle"):
        '''convert lists to ndarray for each field,
        and pickle the resulting dict. duplicates self.history
        in case it needs to be modified as a list later, since ndarray
        conversion is permanent'''
        history = {}
        for field_key, field in self.history.items():
            if not field: continue # skip empty fields
            if isinstance(field, dict): # ["params", "grads", "activations"]
                history[field_key] = {}
                for tensor_key, tensors in field.items():
                    if field_key == "activations":
                        history[field_key][tensor_key] = (
                            torch.cat(tensors)
                            .detach()
                        )
                    elif isinstance(tensors[-1],torch.Tensor):
                        history[field_key][tensor_key] = (
                            torch.stack(tensors)
                            .detach()
                            .numpy()
                        )
                    else:
                        history[field_key][tensor_key] = tensors
            elif isinstance(field, list):
                history[field_key] = field
                
        np.savez(filepath, **history)
        # with open(filepath, mode="wb") as f:
            # pickle.dump(history, f)
    
    def load_history(self, filepath="./logs/history.pickle"):
        with open(filepath, mode="rb") as f:
            return pickle.load(f)
    
    def register_hooks(self, net):
        '''registers all recording hooks given a module (must have submodules)'''
        for module_name, module in net.named_children():
            # register activation hooks (to *submodules* of network)
            self.history["activations"][module_name] = []
            hook = lambda _, __, output, module_name=module_name: self.record_activations(output, module_name)
            module.register_forward_hook(hook)
        for param_name, param in net.named_parameters():
            # register parameter hooks (to *supermodule* of network)
            self.history["params"][param_name] = []
            hook = lambda module, _, __, param_name=param_name: self.record_params(module, param_name)
            net.register_forward_hook(hook)
            # register gradient hooks (to Tensor)
            self.history["grads"][param_name] = []
            hook = lambda grad, param_name=param_name: self.record_grads(grad, param_name)
            param.register_hook(hook)
        self.recording = True
    
    def log(self,message=""):
        if self.logs: 
            print("[LOG] {}".format(message))

    def warning(self,message=""):
        if self.warnings: 
            print("[WARNING] {}".format(message))

    def error(self,message=""):
        if self.errors: 
            print("[ERROR] {}".format(message))

    def addtimestamp(self,job):   
        ''' adds a time stamp based on the job that is performed '''
        jobs = {"new_epoch" : 0,
                "train"     : 1,
                "val"       : 2}
        self.history["timestamps"][0].append(jobs[job])
        self.history["timestamps"][1].append(time() - self.starttime)

    def new_epoch(self):
        ''' prepare the history dictionary for a new epoch by appending a list to each stat'''
        for step in ["train","val"]:
            for key,val in self.history[step].items():
                val.append([])
        # first epoch set the start time
        if len(self.history["timestamps"][0]) == 0:
            self.starttime = time()
        self.addtimestamp("new_epoch")
        
    def earlystop(self, method):
        ''' determine if the training should be stopped early '''
        if method == "ptp": 
            check = abs(np.ptp(self.history["val"]["acc_thresh_total"][-1]))
            if  check <= self.earlystop_epsilon:
                msg = "Range of validation loss total of previous epoch is less than %1.6f: %1.7f"
                print( msg % (self.earlystop_epsilon, check))
                return True
        elif method == "mean":
            # check if running mean of current epoch is not changing
            check = abs(np.mean(self.history["val"]["acc_thresh_total"][-1][:-1]) - self.history["val"]["acc_thresh_total"][-1][-1])
            if len(self.history["val"]["acc_thresh_total"][-1]) >= self.earlystop_patience and check <= self.earlystop_within_epoch_epsilon:
                msg = "Within this epoch, the difference between the validation total loss and the running average of the current iteration is less than %1.6f: %1.7f"
                print( msg % (self.earlystop_epsilon, check))
                return True

            # check if mean of current epoch versus past epochs is not changing 
            window   = np.array(self.history["val"]["acc_thresh_total"][-(self.earlystop_lookback+1):-1])
            if window.shape[0] == 0: 
                return False
            axis  = 0 if len(window.shape) == 1 else 1
            check = abs(np.mean(np.mean(window,axis=axis)) - self.get_avg_stat("val","acc_thresh_total"))
            if  check <= self.earlystop_epsilon:
                msg = "The difference between the average epoch validation total loss, averaged over the past %i epochs, and the current average epoch validation total loss is less than %1.6f: %1.7f"
                print(msg % (self.earlystop_lookback, self.earlystop_epsilon, check))
                return True

        return False

    def update_stats(self, step, stats):
        ''' step is a dictionary key (train/val) and stats contains the statistics exported to history '''
        for key,val in self.history[step].items():
            val[-1].append(stats[key].item())

    def get_avg_stat(self,step,stat):
        ''' return the average of the last epoch stats '''
        return np.mean(self.history[step][stat][-1])

    def stats(self):
        msg =  f"Train " 
        msg += "l[%1.4f, %1.4f, %1.4f] "  % (self.get_avg_stat("train","loss_isr"),       self.get_avg_stat("train","loss_decay"),       self.get_avg_stat("train","loss_total"))
        msg += "a[%1.4f, %1.4f, %1.4f]"   % (self.get_avg_stat("train","acc_thresh_isr"), self.get_avg_stat("train","acc_thresh_decay"), self.get_avg_stat("train","acc_thresh_total"))
        msg += " | " 
        msg += f"Val " 
        msg += "l[%1.4f, %1.4f, %1.4f] "  % (self.get_avg_stat("val","loss_isr"),         self.get_avg_stat("val","loss_decay"),         self.get_avg_stat("val","loss_total"))
        msg += "a[%1.4f, %1.4f, %1.4f]"   % (self.get_avg_stat("val","acc_thresh_isr"),   self.get_avg_stat("val","acc_thresh_decay"),   self.get_avg_stat("val","acc_thresh_total")) 
        return msg

    def progress(self, epoch, itr):
        ''' modification of pbftp (progress bar for the people) from Alex Tuna'''
        nprocessed = (epoch-1)*self.niter + itr
        time_diff  = time() - self.starttime
        if nprocessed == 1: # first processing printer header
            msg = "e/Epochs | file/Files | n/N | Hz | elapsed | remaining"
            if not self.eval_mode:
                msg += "| Train loss[isr,decay,total] acc[isr,decay,total] | Val loss[isr,decay,total] acc[isr,decay,total]"
            print(msg)
        rate = (nprocessed+1)/time_diff 
        msg = "\r"
        msg += "%4i/%i | "            % (epoch,      self.nepoch)
        msg += "%3i/%i | "            % (itr,        self.niter)
        msg += "%6i/%i | "            % (nprocessed, self.ntotal)
        msg += "%2i%% | "             % (100*nprocessed/self.ntotal)
        msg += "%4.2fHz | "           % (rate)
        msg += "%4.2fm elapsed | "    % (time_diff/60)
        msg += "%4.2fm remaining | "  % ((self.ntotal-nprocessed)/(rate*60))
        if not self.eval_mode:
            msg += self.stats()
        print(msg)


