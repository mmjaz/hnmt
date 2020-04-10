#!/bin/bash
moses_scripts=/home/mehdi/mosesdecoder/scripts

#zh_segment_home=/idiap/home/lmiculicich/.cache/pip/wheels/ce/32/de/c2be1db5f30804bc7f146ff698c52963f8aa11ba5049811b0d
#kpu_preproc_dir=/fs/zisa0/bhaddow/code/preprocess/build/bin

max_len=200

#export PYTHONPATH=$zh_segment_home

src=en
tgt=de
pair=$src-$tgt

name=nc2016
nameOfFile=${name}_${src}2${tgt}
echo $nameOfFile


# Tokenise the English part
cat ${nameOfFile}_train_$src.txt | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
$moses_scripts/tokenizer/tokenizer.perl -a -l $src  \
> train.tok.$src

# Tokenise the Deutch part
cat ${nameOfFile}_train_$tgt.txt | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $tgt | \
$moses_scripts/tokenizer/tokenizer.perl -a -l $tgt  \
> train.tok.$tgt

#Segment the Chinese part
#python -m jieba -d ' ' < corpus.$src > corpus.tok.$src 

#
###
#### Clean
#$moses_scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 1 $max_len corpus.retained
###
#

#### Train truecaser and truecase
$moses_scripts/recaser/train-truecaser.perl -model truecase-model.$src -corpus train.tok.$src
$moses_scripts/recaser/truecase.perl < train.tok.$src > train.tc.$src -model truecase-model.$src

#### Train truecaser and truecase
$moses_scripts/recaser/train-truecaser.perl -model truecase-model.$tgt -corpus train.tok.$tgt
$moses_scripts/recaser/truecase.perl < train.tok.$tgt > train.tc.$tgt -model truecase-model.$tgt

#ln -s train.tok.$src  train.tc.$src

#
#  
# dev sets

for devset in dev test; do
  for lang  in $src $tgt; do
      $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < ${nameOfFile}_${devset}_$lang.txt | \
      $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
      $moses_scripts/recaser/truecase.perl -model truecase-model.$lang \
      > ${name^^}.$devset.tc.$lang
    
  done

done


#python /home/mehdi/hanNmt/HAN_NMT/full_source/preprocess.py -train_src /home/mehdi/hanNmt/HAN_NMT/preprocess_TED_zh-en/corpus.tc.zh -train_tgt /home/mehdi/hanNmt/HAN_NMT/preprocess_TED_zh-en/corpus.tc.en -train_doc /home/mehdi/hanNmt/HAN_NMT/preprocess_TED_zh-en/corpus.doc -valid_src /home/mehdi/hanNmt/HAN_NMT/preprocess_TED_zh-en/IWSLT15.TED.dev2010.tc.zh -valid_tgt /home/mehdi/hanNmt/HAN_NMT/preprocess_TED_zh-en/IWSLT15.TED.dev2010.tc.en -valid_doc /home/mehdi/hanNmt/HAN_NMT/preprocess_TED_zh-en/IWSLT15.TED.dev2010.zh-en.doc -save_data /home/mehdi/hanNmt/HAN_NMT/preprocess_TED_zh-en/after_preprocess/IWSLT15.TED -src_vocab_size 30000 -tgt_vocab_size 30000 -src_seq_length 80 -tgt_seq_length 80

