import io, re
raw_data=[]
def ocrformat(raw_data):
    print(type(raw_data))
    #print(raw_data) 
    #raw_data.append(file)
    raw_data=raw_data
  

    refine = [x.strip() for x in [x for x in raw_data if x not in (" ", "\t", "\r")]]

    symbols = [":", "|", ",", ".",]

    refine2 = []

    data = {}

    for r in refine:
        sp = r.split(" ")
        refine2.append(sp)
            
    for r in refine2:
        if r[0].endswith(":"):
            if data.get(r[0][0:-1]):
                data[r[0][0:-1]].append(" ".join(r[1::]))
            else:
                data[r[0][0:-1]] = [" ".join(r[1::])]
        else:
            if data.get(r[0]):
                data[r[0]].append(" ".join(r))
            else:
                data[r[0]] = [" ".join(r)]


    return data 
    #for _ in data:
        #print(_," : ",data[_])
