import numpy as np
import time
import pickle
import sys
sys.path.insert(0, '/opt')

from gadgetron import Gadget
from CNNArt.networks.FullyConvolutionalNetworks.motion.eD_VResFCN_Upsampling_final import fPredict as resFCN_predict


class Py_demo(Gadget):
    def process(self, head, data):
        print(time.asctime(time.localtime(time.time())), ': run py_demo.py')
        with open('data.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        with open('head.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(head, f, pickle.HIGHEST_PROTOCOL)
        result = resFCN_predict(X_test=np.squeeze(data), 
                    sModelPath='/opt/CNNArt/cnnart_trainednets/motion/FCN/cnn_training_info.json', 
                    sOutPath='/opt/Data/Output')
        self.put_next(result, data)
        return 0


