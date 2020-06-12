
"""
Authors: Utkrisht Rajkumar, Subrato Chakravorty, Taruj Goyal, Kaustav Datta
"""


import editdistance

def wer(predict, truth):        
    word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
    wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
    return wer
    
def cer(predict, truth):        
    cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
    return cer


def get_gt_string_from_file(align_file):
    sent = []
    with open(align_file) as f:
        for line in f:
            word = line.strip().split()[-1]
            sent.append(word)
    return ' '.join(sent[1:-1]) 


def find_wer_cer(align_file, pred_str):
    gt_sent = get_gt_string_from_file(align_file)
    wer_eval = wer([pred_str],[gt_sent])
    cer_eval = cer([pred_str],[gt_sent])
    return wer_eval, cer_eval


if __name__=="__main__":
    align_file = "gridcorpus/raw/align/align/swwv7s.align"
    pred = "set black with v seven soon"
    res = find_wer_cer(align_file, pred)
    print("Word Error Rate =",res[0][0])
    print("Character Error Rate =", res[1][0])
