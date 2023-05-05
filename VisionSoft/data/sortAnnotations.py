import os
files = 'Annotations'
fn = os.listdir(files)
new_fn = []
sortlist = []
for i in range(len(fn)):
    sortlist.append(int(fn[i].split('.')[0]))
sortlist.sort()
for i in range(len(fn)):
    new_name = files+'/'+str(i+1001)+'.xml'
    os.rename(files+'/'+str(sortlist[i])+'.xml',new_name)