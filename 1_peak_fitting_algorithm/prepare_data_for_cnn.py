from pathlib import Path
import pandas as pd
import numpy as np
from modules.high_level_fitting import pre_processing_cnn
from scipy.signal import find_peaks, peak_prominences

cnn_dir = Path.cwd().joinpath('data','raw','puregraph_size_comparison','Batch-1')
cnn_files = list(cnn_dir.rglob('*.wdf'))
print(cnn_files)

df_cnn = pd.DataFrame()

for file in cnn_files:
    print(file.name)
    existing_cleaned_files = [f.name for f in list(cnn_dir.rglob('*.pkl'))]
    if any(file.name[:-4] in s for s in existing_cleaned_files):
        df_ready = pd.read_pickle(cnn_dir.joinpath(file.name[:-4]+'.pkl'))
    else:
        df_ready, generic_x = pre_processing_cnn(file,1000,3000)
        df_ready.to_pickle(cnn_dir.joinpath(file.name[:-4]+'.pkl'))
    df_ready['file'] = file.name
    df_cnn = pd.concat([df_ready,df_cnn])

df_cnn = df_cnn[['spectrum','data','file']]
df_cnn.reset_index(inplace=True,drop=True)

def find_peaks_array(y):
    peaks, _ = find_peaks(y, distance=200,prominence=(0.1,1),width=(20,200))
    prominences = peak_prominences(y, peaks)[0]
    number_of_peaks = peaks[prominences>0.1].shape[0]
    return number_of_peaks

t = df_cnn['data'].apply(find_peaks_array)

def clean_zone(y,mn,mx):
    ym = np.mean(y[mn:mx])
    return ym

t2 = df_cnn['data'].apply(lambda y: clean_zone(y,750,1200))
t3 = df_cnn['data'].apply(lambda y: clean_zone(y,1410,1500))

print(df_cnn.shape)
df_cnn = df_cnn.loc[(t==3)& (t2<0.1) & (t3<0.2)]

# print(df_cnn.shape)
df_cnn = df_cnn.loc[(t==3) & (t2<0.1)]


df_cnn.reset_index(inplace=True,drop=True)
print(df_cnn.shape)

df_cnn['material'] = df_cnn['file'].str.extract(r'^([a-zA-Z0-9]*)_')
df_cnn['functionalisation'] = df_cnn['file'].str.extract(r'^[a-zA-Z0-9]*_([a-zA-Z0-9]*)_')
df_cnn['map'] = df_cnn['file'].str.extract(r'([0-9]).wdf$').astype(int)

def make_data_long(group_in):
    vals = group_in['data'].values[0].reshape(1,1900)
    df = pd.DataFrame(vals,columns=np.arange(0,1900),index=[0])
    return df

df_cnn_long = df_cnn.groupby(['material','functionalisation','map','spectrum']).apply(make_data_long).reset_index(drop=False)

df_cnn_long.to_csv(Path.cwd().joinpath('data','processed','puregraph_size_comparison_Y.csv'))