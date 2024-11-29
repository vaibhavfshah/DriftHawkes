import numpy as np
import copy
import torch
import pandas as pd

np.random.seed(43)

def rolling_matrix(x, window_size):
    x = np.array(x)
    x = x.flatten()
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n - window_size + 1, window_size), strides=(stride, stride)).copy()

def alpha_func(i):
    return 2 - 0.1 * i  

def beta_func(i):
    return 3.0 + 0.1*i

def base_intensity_func(i):
    return 0.5 
def gen_events(base, alpha, beta, i):
    t = 0
    events = []

    while True:
        intensity = base + alpha * sum(np.exp(-beta * (t - event)) for event in events)
        next_event_time = -np.log(np.random.uniform()) / intensity

        t += next_event_time
        events.append(t)

        if(len(events)==5000-(250*i)):
          break
    print(events[0])
    return np.array(events)


def get_synthetic(domains):  
    
    timestamps = []
    start_indx, end_indx = None, None
    start_indx = 1
    end_indx = domains+1

    
    for i in range(start_indx, end_indx):
     
      timestamps.append(gen_events(base_intensity_func(i), alpha_func(i), beta_func(i), i))

    i = 0
    timestamps_list = []
    for idx, each_task_checkin in enumerate(timestamps):
        timestamps_list.append(each_task_checkin)
    return timestamps_list

def process_synth(timestamps_list, window_size, device):
    all_domains = timestamps_list 
    
    for i in range(len(all_domains)):
        all_domains[i] = np.array([0, all_domains[i]], dtype=object).reshape(1, 2)
        
    arrays_to_concatenate = []
    ar = None
    for i in range(len(all_domains)):
        arrays_to_concatenate.append(np.copy(all_domains[i][0][-1]))
        
    ar = np.concatenate(arrays_to_concatenate)
    all_np_array = np.array([0, ar], dtype=object).reshape(1, 2)
    
    print(all_np_array.shape)
    for i in range(len(all_domains)):
        for j in range(all_domains[i].shape[0]):
            ts_diff = np.diff(all_domains[i][j][-1])
            all_domains[i][j][-1] = rolling_matrix(ts_diff[ts_diff != 0], window_size+1)

    for j in range(all_np_array.shape[0]):
        ts_diff = np.diff(all_np_array[j][-1])
        all_np_array[j][-1] = rolling_matrix(ts_diff[ts_diff != 0], window_size+1)


    all_arr_X = copy.deepcopy(all_np_array)
    all_arr_y = copy.deepcopy(all_np_array[:,-1])
    for j in range(all_arr_X.shape[0]):
        all_arr_y[j] = all_arr_X[j][-1][:, [-1]]
        all_arr_X[j][-1] = all_arr_X[j][-1][:, :-1]
        
    domains_X = []

    all_X_attr_list = []
    all_X_only_attr_list = []
    for i in range(all_arr_X.shape[0]):
        scalar_tensor = torch.tensor(list(all_arr_X[i][1:-1]))#.to(device).requires_grad_()

        repeated_tensor = scalar_tensor.repeat(all_arr_X[i][-1].shape[0], 1)
        all_X_only_attr_list.append(repeated_tensor)
        
        all_X_attr_list.append(torch.from_numpy(all_arr_X[i][-1]).float())#.to(device).requires_grad_())
    all_only_attr = torch.vstack(all_X_only_attr_list).float().to(device).requires_grad_()

    all_X_attr = torch.cat(all_X_attr_list, dim=0).to(device).requires_grad_()
    all_torch_y = []
    for arr in all_arr_y:
        all_torch_y.append(torch.from_numpy(arr).float())#.to(device).requires_grad_())
    all_y_attr = torch.cat(all_torch_y,dim=0).to(device).requires_grad_()    
    return all_domains, None, None, all_X_attr, all_only_attr, all_y_attr

def map_to_season(month_period):
    season_mapping = {
        "12": "Winter",
        "01": "Winter",
        "02": "Winter",
        "03": "Spring",
        "04": "Spring",
        "05": "Spring",
        "06": "Summer",
        "07": "Summer",
        "08": "Summer",
        "09": "Autumn",
        "10": "Autumn",
        "11": "Autumn",
    }
    year = str(month_period).split('-')[0]
    month_str = str(month_period).split('-')[1]
    
    if month_str == '12':
        next_year = str(int(year) + 1)
        season = season_mapping['12']
    else:
        next_year = year
        season = season_mapping[month_str]

    return f"{next_year}-{season}"

def convert_to_numpy(value):
    if isinstance(value, list):
        return np.array(value)
    return value

def getyelp(device, window_size, city='tucson'):
    df1 = pd.read_json("./data/yelp_dataset/yelp_academic_dataset_checkin.json", lines = True)
    df2 = pd.read_json("./data/yelp_dataset/yelp_academic_dataset_business.json", lines = True)


    merged_df = pd.merge(df1, df2, on='business_id', how='inner')
    merged_df = merged_df[merged_df.city.str.lower() == city]
    

    merged_df.date = merged_df.date.str.split(', ').apply(lambda x: pd.to_datetime(x,format='%Y-%m-%d %H:%M:%S'))
    df_exploded = merged_df.explode('date')
    df_exploded['quarter'] = df_exploded['date'].dt.to_period('Q')
    df_exploded['year'] = df_exploded['date'].dt.to_period('Y')
    df_exploded['month'] = df_exploded['date'].dt.to_period('M')
    df_exploded['season'] = df_exploded['date'].dt.to_period('M').apply(map_to_season)

    df_exploded = df_exploded.groupby(['year', 'business_id', 'latitude', 'longitude', 'stars', 'review_count'], as_index=False).agg({'date': list})#,'city':'city', 'latitude':'latitude', 'longitude':'longitude', 'stars':'stars', 'review_count':'review_count'})
    df_exploded = df_exploded.sort_values(by='date', ascending=True)
    df_exploded = df_exploded.groupby(['year'], as_index=False).agg(list)

    
    numpy_array = df_exploded.to_numpy()
    #list of arrays-> each array is consists of arrays- > each of those is a business specific timestamps + attributes data
    all_domains = [] 
    
    start = 0
    end = numpy_array.shape[0]-1
    
    for j in range(start, end):
        result_array_list = []

        for i in range(1, numpy_array[j].shape[0]-1):
            column_result = np.apply_along_axis(convert_to_numpy, axis=0, arr=numpy_array[j][i])
            result_array_list.append(column_result)
        sublist = []
        for i in range(len(numpy_array[j][-1])):
            sublist.append(np.array(np.apply_along_axis(convert_to_numpy, axis=0, arr=numpy_array[j][-1][i]),))
        tsarr = np.array(sublist, dtype=object)
        result_array_list.append(tsarr)
        # Stack the results vertically
        result_array = np.vstack(result_array_list)
        all_domains.append(result_array.T)
        
    business_dict = {}

    for domain in all_domains:
        for row in domain:
            if row[0] in business_dict:            
                business_dict[row[0]].append(len(row[-1]))
            else:
                business_dict[row[0]] = [len(row[-1])]

    threshold_low = window_size+1    

    nval = end - start
    filtered_dict = {key: value for key, value in business_dict.items() if all(val >= threshold_low and len(value) == nval for val in value)}
    
    filtered_ids = set(filtered_dict.keys())

    for i in range(len(all_domains)):
        indices_to_keep = np.where(np.isin(all_domains[i][:, 0], list(filtered_ids)))
        all_domains[i] = all_domains[i][indices_to_keep]

    lat_mean = np.mean(all_domains[0][:,1])
    lat_std = np.std(all_domains[0][:,1])

    long_mean = np.mean(all_domains[0][:,2])
    long_std = np.std(all_domains[0][:,2])

    stars_mean = np.mean(all_domains[0][:,3])
    stars_std = np.std(all_domains[0][:,3])

    revcount_mean = np.mean(all_domains[0][:,4])
    revcount_std = np.std(all_domains[0][:,4])

    # standardize business specific attributes
    for i in range(len(all_domains)):
        all_domains[i][:,1] -= lat_mean
        all_domains[i][:,1] /= lat_std

        all_domains[i][:,2] -= long_mean
        all_domains[i][:,2] /= long_std

        all_domains[i][:,3] -= stars_mean
        all_domains[i][:,3] /= stars_std

        all_domains[i][:,4] -= revcount_mean
        all_domains[i][:,4] /= revcount_std
        
    business_id_dict = {}

    for array in all_domains[:-1]:
        for business in array:
            
            business_id = business[0]
            if business_id in business_id_dict:
                business_id_dict[business_id][-1] = np.concatenate([business_id_dict[business_id][-1], np.copy(business[-1])])
            else:
                business_id_dict[business_id] = np.copy(business)

    arrays_to_concatenate = list(business_id_dict.values())
    all_np_array = np.vstack(arrays_to_concatenate)
    
    for i in range(len(all_domains)):
        for j in range(all_domains[i].shape[0]):
            final_ts = (all_domains[i][j][-1].astype('datetime64[ns]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            ts_diff = np.diff(final_ts)
            all_domains[i][j][-1] = rolling_matrix(ts_diff[ts_diff != 0], window_size+1)

    for j in range(all_np_array.shape[0]):
        final_ts = (all_np_array[j][-1].astype('datetime64[ns]') - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        ts_diff = np.diff(final_ts)
        
        all_np_array[j][-1] = rolling_matrix(ts_diff[ts_diff != 0], window_size+1)


    all_arr_X = copy.deepcopy(all_np_array)
    all_arr_y = copy.deepcopy(all_np_array[:,-1])
    for j in range(all_arr_X.shape[0]):
        all_arr_y[j] = all_arr_X[j][-1][:, [-1]]
        all_arr_X[j][-1] = all_arr_X[j][-1][:, :-1]
        
    domains_X = []

    all_X_attr_list = []
    all_X_only_attr_list = []
    for i in range(all_arr_X.shape[0]):
        scalar_tensor = torch.tensor(list(all_arr_X[i][1:-1]))#.to(device).requires_grad_()

        repeated_tensor = scalar_tensor.repeat(all_arr_X[i][-1].shape[0], 1)
        all_X_only_attr_list.append(repeated_tensor)
        
        all_X_attr_list.append(torch.from_numpy(all_arr_X[i][-1]).float())#.to(device).requires_grad_())
    all_only_attr = torch.vstack(all_X_only_attr_list).float().to(device).requires_grad_()

    all_X_attr = torch.cat(all_X_attr_list, dim=0).to(device).requires_grad_()
    all_torch_y = []
    for arr in all_arr_y:
        all_torch_y.append(torch.from_numpy(arr).float())#.to(device).requires_grad_())
    all_y_attr = torch.cat(all_torch_y,dim=0).to(device).requires_grad_()
    
    return all_domains, None, None, all_X_attr, all_only_attr, all_y_attr
