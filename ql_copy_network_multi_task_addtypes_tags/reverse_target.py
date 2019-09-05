def reverse(in_dir):
    in_f = open(in_dir,encoding="utf-8")
    out_f = open(in_dir+".reverse","w",encoding="utf-8")
    for line in in_f:
        q, l = line.strip().split("\t")
        l = l.split()
        l.reverse()
        out_f.write(q+"\t"+' '.join(l)+"\n")


reverse(r"D:\data\seq2seq\MSPaD.Merge\MSPaD\data_dir_lower\all_predict\train.txt")
reverse(r"D:\data\seq2seq\MSPaD.Merge\MSPaD\data_dir_lower\all_predict\input.txt")
reverse(r"D:\data\seq2seq\MSPaD.Merge\MSPaD\data_dir_lower\all_predict\validation.txt")