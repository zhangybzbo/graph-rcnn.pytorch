import pickle
import numpy as np
import json

def check_rel(p1, p2, img):
    x1 = (p1[2] + p1[0]) / 2
    x2 = (p2[2] + p2[0]) / 2
    y1 = (p1[3] + p1[1]) / 2
    y2 = (p2[3] + p2[1]) / 2
    wrel = np.abs(x1 - x2) / img[0]
    hrel = np.abs(y1 - y2) / img[1]
    if wrel < 0.15 or hrel < 0.15:
        return True
    else:
        return False

with open('results/predictions_new.pl', 'rb') as f:
    data_obj=pickle.load(f)
    
with open('results/predictions_pred.pl', 'rb') as f:
    data_rel=pickle.load(f)
    
with open('/ssd-playpen/home/zhangyb/VQA/GQA/sceneGraphs/train_sceneGraphs.json') as f:
    train_keys = json.load(f).keys()
with open('/ssd-playpen/home/zhangyb/VQA/GQA/sceneGraphs/val_sceneGraphs.json') as f:
    val_keys = json.load(f).keys()
    
rel_count = 0
rel_pure_count = 0

pure_data_train = {}
pure_data_val = {}
pure_data_test = {}
for key, rels in data_rel.items():
    objs = data_obj[key]
    img_size = (objs['image_widths'], objs['image_heights'])
    datum = []
    for triple in rels:
        obj1 = triple[0]
        obj2 = triple[1]
        rel = triple[2]
        p1 = objs['boxes'][obj1[0]]
        assert obj1[1] == objs['scores'][obj1[0]], (key, obj1[1], objs['scores'][obj1[0]])
        p2 = objs['boxes'][obj2[0]]
        assert obj2[1] == objs['scores'][obj2[0]], (key, obj2[1], objs['scores'][obj2[0]])
        if check_rel(p1, p2, img_size):
            datum.append([obj1, obj2, rel])
    rel_count += len(rels)
    rel_pure_count += len(datum)
    if key in train_keys:    
        pure_data_train[key] = datum
    elif key in val_keys:    
        pure_data_val[key] = datum
    else:
        pure_data_test[key] = datum
    
print('old %d rels, save %d rels' % (rel_count, rel_pure_count))
with open('results/predictions_pred_train.pl', 'wb') as f:
    pickle.dump(pure_data_train, f)
    print('save train %d' % len(pure_data_train))
with open('results/predictions_pred_val.pl', 'wb') as f:
    pickle.dump(pure_data_val, f)
    print('save train %d' % len(pure_data_val))
with open('results/predictions_pred_test.pl', 'wb') as f:
    pickle.dump(pure_data_test, f)
    print('save train %d' % len(pure_data_test))