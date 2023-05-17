import numpy as np
import os
# import ast
import shutil

def get_child(path):
    """
    Get immediate child directories in path.
    """
    return [f.path for f in os.scandir(path)]

"______________________________________________________________________________________________________________"
def copy_dir(src, dst):
    """
    Copy a folder to another directory.
    """
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

"______________________________________________________________________________________________________________"
def cmp_dirs(dir1, dir2):
    """
    True if dir1 and dir2 have the exact same content.
    This works only with txt files.
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



