# lines
img_w = 800
img_h = 64

maxTextLen = 200

# IAM char set
wordCharList = "$ !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# KHATT char set
"""
wordCharList = ['ae', 'exc', 'na', 'bsl', 'to', 'ka', 'col', 'sh', 'n8', 'n5', 'ay', 'dh', 'he', 'laah', 'la', 'th', 'ke', 'equ', 'dbq', 'de', 'aa', 'se', 'teE', 'laae', 'bro', 'n0', 'laaa', 'wl', 'n7', 'per', 'ta', 'n9', 'tee', 'qts', 'ha', 'n3', 'ja', 'sa', 'za', 'ah', 'ya', 'hyp', 'wa', 'com', 'fsl', 'ee', 'sp', 'da', 'brc', 'laam','equ', 'exc', 'bsl', 'scr', 'n4', 'zha', 'n6', 'kh', 'am', 'hh', 'fa', 'n1', 'n2', 'gh', 'ba', 'dot', 'al', 'ma', 'ra']
"""

vocab = list(wordCharList)
num_classes = len(wordCharList) + 1

max_to_keep = 10

lr = 1e-3
wd = 1e-4

epochs = 300
batch_size = 32

best_cer = 1000.0
best_ctc = 1000.0

weight_path = "./output/weights/iam"
model_name = "model_gmlp_4"
train_path = "data/IAM/train.txt"
test_path = "data/IAM/test.txt"
validation_path = "data/IAM/val.txt"

use_gMlp_only = True
L = 4
gMlp_units = 512
gMLP_dropout = 0.01

use_lstm = True
rnn_units = 256
