

def multiLine2oneLine(inputfile, outputfile1, outputfile2):
    content = []
    with open(inputfile, 'r') as f:
        for line in f:
            content.append([int(val) for val in line.strip().split()])


    with open(outputfile1, 'w') as f:
        f.write("1\n")
        for line in content:

            f.write('{} '.format(" ".join(str(val) for val in line)))

    with open(outputfile2, 'w') as f:

        for line in content:

            f.write('{} '.format(" ".join(str(val) for val in line)))

input = "../output/citeseerMultiLine_unheader.dat"
output1 = "../output/citeseerOneLine.dat"
output2 = "../output/citeseerOneLine_unheader.dat"
multiLine2oneLine(input, output1, output2)