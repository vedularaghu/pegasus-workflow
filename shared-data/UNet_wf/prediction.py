import h5py
import pickle
import os
from tensorflow.keras.models import load_model

CURR_PATH = os.getcwd()

infile = open(CURR_PATH + "/data_split.pkl",'rb')

new_dict = pickle.load(infile)

infile.close()

path = CURR_PATH

test_data = new_dict['test']

X_test = [cv2.imread(os.path.join(path,i))[:,:,0] for i in test_data]

model = load_model(CURR_PATH+"/model.h5", compile=False)

test_vol = np.array(X_test, dtype=np.float32)

preds = model.predict(test_vol)

for i in range(len(preds)):
    img = np.squeeze(preds[pred_candidates[i]])
    cv2.imwrite(str(test_data[i].split('.png')[0]+'_mask.png'), img)