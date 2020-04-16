import pickle, h5py

h5fille = 'datasets/VG_GQA/GQA_imdb_1024.h5'
resultfile = 'results/predictions.pl'
with open(resultfile, 'rb') as f:
    data = pickle.load(f)

h5f = h5py.File(h5fille, 'r')
for i, ids in enumerate(h5f['image_ids']):
    index = ids.decode("utf-8")
    data[index]['image_heights'] = h5f['image_heights'][i]
    data[index]['image_widths'] = h5f['image_widths'][i]
    data[index]['original_heights'] = h5f['original_heights'][i]
    data[index]['original_widths'] = h5f['original_widths'][i]
    
newfile = 'results/predictions_new.pl'
with open(newfile, 'wb') as f:
    pickle.dump(data, f)