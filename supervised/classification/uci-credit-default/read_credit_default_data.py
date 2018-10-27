import pandas as panda

remote_location = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

def downLoadData():

    data = panda.read_excel(remote_location,sheet_name = "Data", header = 1)

    data.rename(str.lower, inplace = True, axis = 'columns')

    print(data.dtypes)

    return data

downLoadData()