# @Author: nfrazee
# @Date:   2018-08-17T10:43:38-04:00
# @Last modified by:   nfrazee
# @Last modified time: 2018-10-11T09:32:48-04:00
# @Comment: Analyze a file in a 2d space of bins


# These are both x and y bin values (min must be less than max)
bin_min=0
bin_max=300
bin_num=30
# This is the name of the file you have your data in
in_file_name="all11.txt"
# This is the name of the file you want the analyzed data to go to
out_file_name="out.dat"
# The number of the column for X (space-separated, starting at 1)
x_col=6
# The number of the column for Y (space-separated, starting at 1)
y_col=7
# The number of the column for Z (space-separated, starting at 1)
z_col=11

########################################################
###### End User input section! Read on for fun! :D #####
########################################################

bin_width=(bin_max-bin_min)/bin_num
bin_center=bin_width/2
# Will contain the bins for the data
bin_list=[]
# Will contain the data from the file
data_list=[]

def bin_list_maker(min,max,num,binList):
    """Makes a list of bins with centers based on the values provided"""

    # Checks your min and max make sense
    if min >= max:
        raise ValueError("Your bin_max must be greater than your bin_min!")

    bin_width=(max-min)/num

    x_count=1
    # Makes bins within range
    for x in range(0,num):
        y_count=1
        for y in range(0,num):
            # Bin composition is: X value, Y value, Total value, Number of values
            binList.append([min+(x_count*bin_width),min+(y_count*bin_width),0,0])
            y_count+=1
        x_count+=1
    return None

def open_file(fileName,listGiven):
    """Opens the file and appends the important data to a list"""


    # Check if the files exist
    try:
        filein=open(str(fileName),'r')
    except IOError:
        print("File not found!")

    for x in filein:
        # Looks for lines begining with ATOM to avoid the processing of blank space
        if x[0] == "A":
            # Appends only the user specified data to the list of data
            listGiven.append([x.split()[x_col-1],x.split()[y_col-1],x.split()[z_col-1]])

    filein.close()

    return None


def evaluate(listGiven,binList,binMin):
    """This fuction iterates through the given list calculating the probability for each bin then counting up the CDF"""
    # Normalizes the data points to be added with each iteration so that they add to 1
    normalized=1/(len(listGiven))
    # Evaluates each data point
    for g in listGiven:
        # Iterate through the bin list
        for b in binList:
            # Check if its less than the x bin
            if float(g[0]) <= float(b[0]):
                # Check if its less than the y bin
                if float(g[1]) <= float(b[1]):
                    # Add the value of the data_list to the total value for that bin
                    b[2]+=float(g[2])
                    # Add one to the total number of values
                    b[3]+=1
                    # Once the point is counted once the break will keep it from being counted again
                    break



def writer(binList,fileName):
    """Writes out a file with the data from your bin list"""
    outfile=open(str(fileName),'w')
    outfile.write("# X Bin Center     Y Bin Center      Average Value        Probability \n")

    count=0
    for x in binList:
        # Avoids division by zero error for unpopulated bins
        if float(binList[count][3]) != 0:
            mean=str(float(binList[count][2])/float(binList[count][3]))
        else:
            mean="0"
        # The data is as follows: X Bin Center      Y Bin Center       Average value for bin       Probability the bin is occupied
        outfile.write(str(binList[count][0]-bin_center)+"    "+str(binList[count][1]-bin_center)+"     "+mean+"    "+str(float(binList[count][3])/len(data_list))+"\n")
        count+=1

    outfile.close
    return None

def main():
    bin_list_maker(bin_min,bin_max,bin_num,bin_list)

    open_file(in_file_name,data_list)

    evaluate(data_list,bin_list,bin_min)

    writer(bin_list,out_file_name)

    return None

if __name__ == "__main__":
    main()
