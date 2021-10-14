# @Author: nfrazee
# @Date:   2019-03-28T10:00:30-04:00
# @Last modified by:   nfrazee
# @Last modified time: 2019-03-28T11:40:43-04:00
# @Comment:


import sys

in_name=str(sys.argv[1])
out_name=str(sys.argv[2])

bin_min=int(sys.argv[3])
bin_max=int(sys.argv[4])
bin_num=int(sys.argv[5])
data_col=int(sys.argv[6])

data_list=[]
bin_list=[]

def bin_list_maker(min,max,num,binList):
    """Makes a list of bins with centers based on the values provided"""

    bin_width=(max-min)/num

    count=1
    # Makes bins within range
    for x in range(0,num):
        binList.append([min+(count*bin_width),0,0])
        count+=1

    return None



def open_file(listGiven):
    """Opens the files in the list of directories"""

    # Check if the files exist
    try:
        filein=open(in_name,'r')
    except IOError:
        print("file not found")


    for x in filein:
        if x[0] != "#":
            listGiven.append(x.split())

    filein.close()

    return None

def evaluate(listGiven,binList):
    """This fuction iterates through the given list calculating the probability for each bin then counting up the CDF"""
    # Normalizes the data points to be added with each iteration so that they add to 1
    normalized=1/(len(listGiven))
    # Evaluates each data point
    for x in listGiven:
        # Iterate through the bin list
        for b in binList:
            # Check if its less than that bin
            if float(x[data_col]) < float(b[0]):
                # Count up 1 but normalize it
                b[1]+=normalized
                # Once the point is counted once the break will keep it from being counted again
                break

    # Counting up the CDF
    count=0
    # Iterates through the binList
    for i in binList:
        # If this is the start
        if count==0:
            # Set the CDF to the PDF
            i[2]=i[1]
        else:
            # Count up the previous value and the current PDF
            i[2]=binList[count-1][2]+i[1]

        count+=1


def writer(binList):
    """Writes out a file with the data from your bin list"""
    outfile=open(out_name,'w')
    outfile.write("# Bin Center         PDF            CDF\n")
    bin_width=(bin_max-bin_min)/bin_num
    bin_center=bin_width/2
    count=0
    for x in binList:
        outfile.write(str(binList[count][0]-bin_center)+"    "+str(binList[count][1])+" "+str(binList[count][2])+"\n")
        count+=1

    outfile.close
    return None

def main():
    bin_list_maker(bin_min,bin_max,bin_num,bin_list)

    open_file(data_list)

    evaluate(data_list,bin_list)

    writer(bin_list)

    return None

if __name__ == "__main__":
    main()
