def unpack(data):
    for i, arr in enumerate(data):
        data[i] = arr[0][0]
    return data

def padding(lst, max_len):
    return lst + [0] * (max_len - len(lst))
