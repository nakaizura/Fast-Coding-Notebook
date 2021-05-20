with open('./train_img_feats.pkl', 'rb') as f:
    train_img_feats = pickle.load(f, encoding='latin1')  # 20757 2048 // 2281 2048
    train_img_feat = np.array(train_img_feats)
with open('./train_txt_vecs.pkl', 'rb') as f:
    train_txt_vecs = pickle.load(f, encoding='latin1')  # 20757 1024
with open('./train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f, encoding='latin1')  # 20757
with open('./test_img_feats.pkl', 'rb') as f:
    test_img_feats = pickle.load(f, encoding='latin1')  # 19017 2048 //13591 2048
with open('./test_txt_vecs.pkl', 'rb') as f:
    test_txt_vecs = pickle.load(f, encoding='latin1')  # 19017 1024
with open('./test_labels.pkl', 'rb') as f:
    test_labels = pickle.load(f, encoding='latin1')  # 19017

print('download over')

state = np.random.get_state()
np.random.shuffle(train_img_feats)
np.random.set_state(state)
np.random.shuffle(train_txt_vecs)
np.random.shuffle(train_labels)

state2 = np.random.get_state()
np.random.shuffle(test_img_feats)
np.random.set_state(state2)
np.random.shuffle(test_txt_vecs)
np.random.shuffle(test_labels)

print('shuffle over')

with open('./train_img_feats.pkl', 'wb') as f:
    pickle.dump(train_img_feats, f, pickle.HIGHEST_PROTOCOL)
with open('./train_txt_vecs.pkl', 'wb') as f:
    pickle.dump(train_txt_vecs, f, pickle.HIGHEST_PROTOCOL)
with open('./train_labels.pkl', 'wb') as f:
    pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)
with open('./test_img_feats.pkl', 'wb') as f:
    pickle.dump(test_img_feats, f, pickle.HIGHEST_PROTOCOL)
with open('./test_txt_vecs.pkl', 'wb') as f:
    pickle.dump(test_txt_vecs, f, pickle.HIGHEST_PROTOCOL)
with open('./test_labels.pkl', 'wb') as f:
    pickle.dump(test_labels, f, pickle.HIGHEST_PROTOCOL)
