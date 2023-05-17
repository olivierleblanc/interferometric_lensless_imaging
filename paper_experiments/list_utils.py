import numpy as np
import os
import ast
import shutil

"_______________________________________________________________"
def copy_dir(src, dst):
  h = os.getcwd()
  src = r"{}".format(src)
  if not os.path.isdir(dst):
     print("\n[!] No Such directory: ["+dst+"] !!!")
     exit(1)

  if not os.path.isdir(src):
     print("\n[!] No Such directory: ["+src+"] !!!")
     exit(1)
  if "\\" in src:
     c = "\\"
     tsrc = src.split("\\")[-1:][0]
  else:
    c = "/"
    tsrc = src.split("/")[-1:][0]
    
  os.chdir(dst)
  if os.path.isdir(tsrc):
    print("\n[!] The Directory already exists.")
    exit(1)
  
  if (bool(tsrc.strip())):
    os.mkdir(tsrc)
  os.chdir(h)
  files = []
  for i in os.listdir(src):
    files.append(src+c+i)
  if len(files) > 0:
    for i in files:
        if not os.path.isdir(i):
            shutil.copy2(i, dst+c+tsrc)

  print("\n[*] Done ! :)")

"_______________________________________________________________"
def flatten(thelist, flat_list=[]):
    """
    Flattens a multidimensional array into a 1D array 
    """
    for elem in thelist:
        if isinstance(elem, list):
            flatten(elem, flat_list)
        else :
            flat_list.append(elem)

    return flat_list

"_______________________________________________________________"

def init_extensible_list(theshape):
    return np.ones(theshape+(1,)).astype('int').tolist()

"_______________________________________________________________"

def add_in_list(pos, val, thelist, new0=False):
    if (pos[0]>len(thelist)-1):
        thelist.append([[val]])
    else:
        if (new0):
            thelist.insert(pos[0], [[val]])
        elif (pos[1]>len(thelist[pos[0]])-1):
            thelist[pos[0]].append([val])
        else:
            thelist[pos[0]].insert(pos[1], [val])

"_______________________________________________________________"

def addval2list_2D (thelist, pos, elems):
    for elem in elems:
        thelist[pos[0]][pos[1]].append(elem)

"_______________________________________________________________"

def write_list2txt (path, thelist):
    with open(path, 'w') as f:
        L = len(thelist)
        f.write('[')
        for i, item in enumerate(thelist):
            f.write(str("{}".format(item)))
            if (i<L-1):
                f.write(", ")
        f.write(']')
        f.close()

"_______________________________________________________________"

def read_list_from_txt (path):
    with open(path, 'rb') as f:
        stringlist = f.read().decode("utf-8") 
        f.close()
        return ast.literal_eval(stringlist)

"_______________________________________________________________"

def init_inputs_mat (vals1, vals2):
    thelist =[]
    for i in vals1:
        small = []
        for j in vals2:
            small.append([i,j])
        thelist.append(small)

    return thelist

"_______________________________________________________________"

def search_in_inputs(pair, thelist):
    x,y = pair
    if (len(thelist[0][0])==0):
        # If the list is empty
        return None
    for i in range(len(thelist)):
        for j in range(len(thelist[i])):
            elemx, elemy = thelist[i][j]
            if (x>elemx):
                break
            elif(x==elemx):
                if (y==elemy):
                    return (i,j)          
            else:
                return None
    return None

"_______________________________________________________________"

def add_in_inputs(pair, thelist):
    new0 = False

    if (len(thelist[0][0])==0):
        # If the list is empty
        thelist[0][0] = [pair[0], pair[1]]
        return new0, (0,0)

    x,y = pair
    i=0
    j=0
    for i in range(len(thelist)):
        for j in range(len(thelist[i])):
            # print(i,j)
            elemx, elemy = thelist[i][j]
            if (x>elemx):
                break       
            elif (x==elemx):
                if (y<elemy):
                    thelist[i].insert(j, [pair[0], pair[1]]) 
                    return new0, (i,j)

                if (j==len(thelist[i])-1):
                    thelist[i].append([pair[0], pair[1]])
                    return new0, (i,j+1)
            else:
                thelist.insert(i, [[pair[0], pair[1]]])
                new0 = True
                return new0, (i,0)
    thelist.append([[pair[0], pair[1]]])  

    return new0, (i+1,j) 

"________________________________________________________________"

def add_normdata2txt(path, data, pos):
    norms = read_list_from_txt(path)
    addval2list_2D (norms, pos, [data])
    write_list2txt (path, norms)

"________________________________________________________________"

def save_data2prefix(fixed_val, possibilities, variables, prefix, norms, visib_card):

    if (os.path.exists(prefix) is False):
        os.makedirs(prefix, exist_ok=True)

    vary_val_str = "".join([val for val in possibilities if val!=fixed_val])
    path_inputs = prefix+vary_val_str+'_values.txt'
    path_GTnorms = prefix+'GTnorms.txt'
    path_errnorms = prefix+'errnorms.txt'
    path_intM_GTnorms = prefix+'intM_GTnorms.txt'
    path_intM_errnorms = prefix+'intM_errnorms.txt'
    path_visibilities_cardinality = prefix+'visib.txt'

    pair=()
    for i,elem in enumerate(variables):
        if (possibilities[i]!=fixed_val):
            pair+=(elem,)

    paths = [path_GTnorms, path_errnorms, path_intM_GTnorms, path_intM_errnorms]

    # # If the files don't exist yet, create and instantiate them
    # if (os.path.exists(path_inputs) is False):
    #     write_list2txt (path_inputs, init_inputs_mat ([pair[0]], [pair[1]]))
    #     for i, path in enumerate(paths):
    #         tmp = init_extensible_list((1,1))
    #         tmp2 = [[[ z*norms[i] for z in y] for y in x] for x in tmp]
    #         write_list2txt (path, tmp2)
    # # Otherwise, insert data in the files
    # else:

    "The files should have been created in choose_pair_tot"
    thelist = read_list_from_txt (path_inputs)
    pos = search_in_inputs(pair, thelist)
    if (pos is None):
        new0, pos = add_in_inputs(pair, thelist)
        write_list2txt (path_inputs, thelist)

        for i, path in enumerate(paths):
            mat = read_list_from_txt (path)
            add_in_list(pos, norms[i], mat, new0)
            write_list2txt (path, mat)
        mat = read_list_from_txt (path_visibilities_cardinality)
        add_in_list(pos, visib_card, mat, new0)
        write_list2txt (path_visibilities_cardinality, mat)
    else:
        for i, path in enumerate(paths):
            add_normdata2txt (path, norms[i], pos)
        add_normdata2txt (path_visibilities_cardinality, visib_card, pos)

"____________________________________________________________________"

def save_data(fixed_val, possibilities, variables, folder_path, norms, visib_card):

    prefix = folder_path+r'{}_fixed/{}/'.format(fixed_val, variables[np.where([elem==fixed_val for elem in possibilities])[0][0]])
    if (os.path.exists(prefix) is False):
        os.makedirs(prefix, exist_ok=True)

    save_data2prefix(fixed_val, possibilities, variables, prefix, norms, visib_card)

"_____________________________________________________________________"
def cmp_dirs(dir1, dir2):
    """True if dir1 and dir2 have the exact same content
    """
    files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

    for i in range(len(files1)):
        # reading files
        f1 = open(files1[i], "r")  
        f2 = open(files2[i], "r")  

        for line1 in f1:
            for line2 in f2:  
                # matching line1 from both files
                if line1 != line2:  
                    f1.close()                                       
                    f2.close()   
                    return False      
        f1.close()                                       
        f2.close()   
    return True

"______________________________________________________________________"
def is_in_dir(dir1, dir2):
    """True if content of dir1 is found entirely in dir2 
    """
    files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

    for i in range(len(files1)-1):
        # reading files
        list1 = read_list_from_txt (files1[i])
        list1 = flatten(list1, [])
        list2 = read_list_from_txt (files2[i])
        list2 = flatten(list2, [])
        
        for j, elem in enumerate(list1):
            if (any(elem for elem2 in list2) is False):
                return False
    return True

"_____________________________________________________________________"
def is_in_pairs(dir1, dir2):
    """True if the pairs of values of dir1 are found entirely in dir2 
    """
    files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

    i = len(files1)-1
    # reading files
    list1 = read_list_from_txt (files1[i])
    list2 = read_list_from_txt (files2[i])
    
    for j, elem in enumerate(list1):
        for elemnew in elem:
            if (any([elemnew for elem2new in elem2] for elem2 in list2) is False):
                return False
    return True

"_____________________________________________________________________"
def append_dirs(dir1, dir2, dest, fixed_val, possibilities):

    vary_val_str = "".join([val for val in possibilities if val!=fixed_val])
    path_inputs1 = dir1+vary_val_str+'_values.txt'
    path_GTnorms1 = dir1+'GTnorms.txt'
    path_errnorms1 = dir1+'errnorms.txt'
    path_intM_GTnorms1 = dir1+'intM_GTnorms.txt'
    path_intM_errnorms1 = dir1+'intM_errnorms.txt'
    paths1 = [path_GTnorms1, path_errnorms1, path_intM_GTnorms1, path_intM_errnorms1]
    
    if (dir2 != dest):
      copy_dir(dir2, dest)

    path_inputs2 = dest+vary_val_str+'_values.txt'
    path_GTnorms2 = dest+'GTnorms.txt'
    path_errnorms2 = dest+'errnorms.txt'
    path_intM_GTnorms2 = dest+'intM_GTnorms.txt'
    path_intM_errnorms2 = dest+'intM_errnorms.txt'
    paths2 = [path_GTnorms2, path_errnorms2, path_intM_GTnorms2, path_intM_errnorms2]

    thelist1 = read_list_from_txt (path_inputs1)
    thelist2 = read_list_from_txt (path_inputs2)

    for i in range(len(thelist1)):
        for j in range(len(thelist1[i])):
            pair1 = thelist1[i][j]

            pos2 = search_in_inputs(pair1, thelist2)
            if (pos2 is None):
                new0, pos2 = add_in_inputs(pair1, thelist2)
                write_list2txt (path_inputs2, thelist2)

                for k, path2 in enumerate(paths2):
                    mat1 = read_list_from_txt (paths1[k])
                    mat2 = read_list_from_txt (path2)
                    norm1 = mat1[i][j][0]
                    add_in_list(pos2, norm1, mat2, new0)
                    addval2list_2D(mat2, pos2, mat1[i][j][1:])
                    write_list2txt (path2, mat2)
            else:
                for k, path2 in enumerate(paths2):
                    mat1 = read_list_from_txt (paths1[k])
                    for norm1 in mat1[i][j]:
                        add_normdata2txt (path2, norm1, pos2)

"______________________________________________________________________"

def count_trials(allpairs, inputs, GTnorms):
    pair_posNmult = []

    for i,elem in enumerate(allpairs):
        for j,elem2 in enumerate(elem):
            pos = search_in_inputs(elem2, inputs)
            if (pos is None):
                pair_posNmult += [(elem2, 0, 0)]
            else:
                pair_posNmult += [(elem2, pos, len(GTnorms[pos[0]][pos[1]]))]
    return pair_posNmult

"______________________________________________________________________"
def choose_pair(pairsNpos, ntrial):
    for pair in pairsNpos:
        if pair[2]<ntrial:
            Q,M = pair[0]
            return (Q,M, pair[2])
    return None,None,None

"______________________________________________________________________"
def get_pairs(Qs,Ms):
    thelist=[[[]]]
    for Q in Qs:
        for M in Ms:
            pair = (Q,M)
            add_in_inputs(pair, thelist)
    return thelist

"______________________________________________________________________"
def choose_pair_tot(prefix, fixed_val, possibilities, thelist, ntrial):
    vary_val_str = "".join([val for val in possibilities if val!=fixed_val])
    path_inputs = prefix+vary_val_str+'_values.txt'
    path_GTnorms = prefix+'GTnorms.txt'

    # Check if the directory exists
    if (os.path.exists(prefix) is False):
        # If not, instantiate the files
        os.makedirs(prefix, exist_ok=True)
        path_errnorms = prefix+'errnorms.txt'
        path_intM_GTnorms = prefix+'intM_GTnorms.txt'
        path_intM_errnorms = prefix+'intM_errnorms.txt'
        path_visibilities_cardinality = prefix+'visib.txt'
        write_list2txt (path_inputs, [[[]]])
        write_list2txt (path_GTnorms, [[[]]])
        write_list2txt (path_errnorms, [[[]]])
        write_list2txt (path_intM_GTnorms, [[[]]])
        write_list2txt (path_intM_errnorms, [[[]]])
        write_list2txt (path_visibilities_cardinality, [[[]]])

    inputs = read_list_from_txt (path_inputs)
    GTnorms = read_list_from_txt(path_GTnorms)
    pairsNpos = count_trials(thelist, inputs, GTnorms)
    return choose_pair(pairsNpos, ntrial)
