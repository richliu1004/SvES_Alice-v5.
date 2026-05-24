
def txt2list(fp_txt):
    txt=[]
    content=[]
    f=open(fp_txt,'r')
    for i in f.readlines():
        content.append(i)
    f.close()
    content[0]=content[0].split('\n')[0]
    for i in range(len(content)):
        content[i]=content[i].split(' ')
        for j in ['','0','2','\n']:
            while True:
                try:
                    content[i].remove(j)
                except:
                    break
    txt=content
    for i in range(len(txt)):
        for j in range(len(txt[i])):
            txt[i][j]=float(txt[i][j])
    return txt

def list2txt(list,output):
    lines=[]
    f=open(output,'w')
    for i in range(len(list)):
        bbox=list[i][0:4]
        kypt=[]
        for j in range(0,len(list[i][4:]),2):
            k=list[i][4+j:4+j+2]+['2']
            kypt.extend(k)
        l=[str(x)+' ' for x in [0]+bbox]+[str(x)+' ' for x in kypt]+['\n']
        lines.extend(l)
    f.writelines(lines)
    f.close()
    return lines

if __name__=='__main__':
    # print(len(txt2list(r'C:\Users\LIU\Desktop\test_funcMOD\MP\label\txt\0.txt')))
    # print(len(txt2list(r'C:\Users\LIU\Desktop\test_funcMOD\KP\label\txt\0.txt')))

    print(txt2list(r'C:\Users\LIU\Desktop\test_funcMOD\MP\label\txt\0.txt'))
    # print(list2txt(txt2list(r'C:\Users\LIU\Desktop\test_funcMOD\KP\label\txt\0.txt'),r'C:\Users\LIU\Desktop\temp.txt'))
