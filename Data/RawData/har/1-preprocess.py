import numpy as np

datasets = ["hhar", "uci", "motion", "shoaib"]
# users set as device-side data
user_device_dic = {"hhar": [4,5,9], "uci": [10,11,12,13,14], 
                "motion": [39,40,41,42,43], "shoaib":[63,64]} 
data = np.load("./merge/data_20_120.npy", allow_pickle=True) # [sample_num, window_size, feature_num]
label = np.load("./merge/label_20_120.npy", allow_pickle=True) # [activity, user, domain]
        
# Reshape data and labels
x = data.reshape(data.shape[0]*data.shape[1]//20, 20, data.shape[2]) # [sample_num, window_size=20, feature_num=6]
label = label.reshape(label.shape[0]*label.shape[1]//20, 20, label.shape[2])
        
x, y, user, device = x, label[:, :, 0].sum(1)//20, label[:, :, 1].sum(1)//20, label[:, :, 2].sum(1)//20
print('Shape:\tData: {}\tLabel: {}\tUser: {}\tDevice: {}'.format(x.shape, y.shape, user.shape, device.shape))
        
# Process data
for i, dataset in enumerate(datasets):
    idx = np.where(device==i)
    x_tmp, y_tmp, user_tmp = x[idx], y[idx], user[idx]
    user_device = user_device_dic[dataset] # user IDs set as device-side data
    user_cloud = np.setdiff1d(np.unique(user_tmp), user_device) # user IDs set as cloud-side data
            
    x_tmp_device = np.concatenate([x_tmp[np.where(user_tmp==j)] for j in user_device], 0)
    y_tmp_device = np.concatenate([y_tmp[np.where(user_tmp==j)] for j in user_device], 0)
    user_tmp_device = np.concatenate([user_tmp[np.where(user_tmp==j)] for j in user_device], 0)
    print('Device Data:\t', x_tmp_device.shape, y_tmp_device.shape)
    np.savez("{}_device.npz".format(dataset), x_tmp_device, y_tmp_device, user_tmp_device)
                        
    x_tmp_cloud = np.concatenate([x_tmp[np.where(user_tmp==j)] for j in user_cloud], 0)
    y_tmp_cloud = np.concatenate([y_tmp[np.where(user_tmp==j)] for j in user_cloud], 0)
    user_tmp_cloud = np.concatenate([user_tmp[np.where(user_tmp==j)] for j in user_cloud], 0)
    print('Cloud Data:\t', x_tmp_cloud.shape, y_tmp_cloud.shape)
    np.savez("{}_cloud.npz".format(dataset), x_tmp_cloud, y_tmp_cloud, user_tmp_cloud)