# trans-ner
Named Entity Recognition with [Transformers](https://github.com/huggingface/transformers)

## Prepare

```bash
git clone https://github.com/ivlcic/trans-ner.git
cd trans-ner
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Since corpora is small, we keep converted and split corpora in data directory, but the prep step can be repeated:
```
src/prep.py [lang]
```

We keep 80/10/10 corpora splits in `CVS` format with two fields:

- sentence: space separated word list representing a sentence
- NER: 

Prepared files are also kept for possible manual check.
We dropped sentences longer than 128 word tokens since there are less than 20 of them in any corpus.

## Training

Custom models can be trained with custom corpora selection:

```bash
# train on cuda device 0 with learning rate 2e-5 for 40 epochs and batch size of 20
# use sl_bsnlp hr_500k sr_set corpora for fine tuning with XLM Roberta Base pretrained model
# exclude MISC NER tags    
src/train.py -c 0 -l 2e-5 -e 40 -b 20 --no_misc sl_bsnlp hr_500k sr_set xlmrb &> test-xlmrb-nomisc.log
```

```bash
# train on cuda device 0 with learning rate 2e-5 for 40 epochs and batch size of 20
# use all Slovene, Croatian and Serbian corpora for fine tuning with XLM Roberta Base pretrained model   
src/train.py -c 0 -l 2e-5 -e 40 -b 20 sl hr sr &> xlmrb-sl.hr.sr.log
```

Note: 

- Result model names (target directory) are derived from pretrained model, corpora.
It might change in th future.
- We limited transformers tokenization to max length of 256, but this can be changed with training flags.
- We also provided other flags, but we kept them out of intro page for brevity. 

## Testing
Test the fine-tuned model with test part of corpora
```
# run evaluation on cuda device 1 with best checkpoint the training kept
src/test.py -c 1 sl hr sr xlmrb-sl.hr.sr/checkpoint-39936
```

## Inference
You can do inference if needed for manual check. 
```
src/infer.py -c 0 xlmrb-sl_comb/checkpoint-39402 sl 'Po prejeti podpori kolegov vrhovnih \
sodnikov, je kandidat za predsednika vrhovnega sodišča Miodrag Đorđević včeraj zvečer \
prejel tudi podporo sodnega sveta. Odločanje sodnega sveta naj bi bilo sicer "napeto".'
```


# Used corpora

We keep used NER corpora in this repository just for convenience.

## Slovene 

- [Training corpus SUK 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1747)
  
```
@misc{11356/1747,
 title = {Training corpus {SUK} 1.0},
 author = {Arhar Holdt, {\v S}pela and Krek, Simon and Dobrovoljc, Kaja and Erjavec, Toma{\v z} and Gantar, Polona and {\v C}ibej, Jaka and Pori, Eva and Ter{\v c}on, Luka and Munda, Tina and {\v Z}itnik, Slavko and Robida, Nejc and Blagus, Neli and Mo{\v z}e, Sara and Ledinek, Nina and Holz, Nanika and Zupan, Katja and Kuzman, Taja and Kav{\v c}i{\v c}, Teja and {\v S}krjanec, Iza and Marko, Dafne and Jezer{\v s}ek, Lucija and Zajc, Anja},
 url = {http://hdl.handle.net/11356/1747},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {Creative Commons - Attribution-{NonCommercial}-{ShareAlike} 4.0 International ({CC} {BY}-{NC}-{SA} 4.0)},
 issn = {2820-4042},
 year = {2022} 
}
```
- [BSNLP: 3rd Shared Task on SlavNER](http://bsnlp.cs.helsinki.fi/shared-task.html)
  We merged 2017+2021 train data with 2021 test data and made custom train / dev / test splits. 
  We also mapped EVT (event) and PRO (product) tags to MISC to align the corpus with others.
  You can change mappings running a custom prepare corpus step (see above).

## Croatian

- [Training corpus hr500k 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1183)

```
@misc{11356/1183,
 title = {Training corpus hr500k 1.0},
 author = {Ljube{\v s}i{\'c}, Nikola and Agi{\'c}, {\v Z}eljko and Klubi{\v c}ka, Filip and Batanovi{\'c}, Vuk and Erjavec, Toma{\v z}},
 url = {http://hdl.handle.net/11356/1183},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {Creative Commons - Attribution-{ShareAlike} 4.0 International ({CC} {BY}-{SA} 4.0)},
 issn = {2820-4042},
 year = {2018} 
}
```

## Serbian

- [Training corpus SETimes.SR 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1200)

```
@misc{11356/1200,
 title = {Training corpus {SETimes}.{SR} 1.0},
 author = {Batanovi{\'c}, Vuk and Ljube{\v s}i{\'c}, Nikola and Samard{\v z}i{\'c}, Tanja and Erjavec, Toma{\v z}},
 url = {http://hdl.handle.net/11356/1200},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {Creative Commons - Attribution-{ShareAlike} 4.0 International ({CC} {BY}-{SA} 4.0)},
 issn = {2820-4042},
 year = {2018} 
}
```
# Evaluation

For evaluation, we use [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval)
```
@misc{seqeval,
  title={{seqeval}: A Python framework for sequence labeling evaluation},
  url={https://github.com/chakki-works/seqeval},
  note={Software available from https://github.com/chakki-works/seqeval},
  author={Hiroki Nakayama},
  year={2018},
}
```

Which is based on
```
@inproceedings{ramshaw-marcus-1995-text,
    title = "Text Chunking using Transformation-Based Learning",
    author = "Ramshaw, Lance  and
      Marcus, Mitch",
    booktitle = "Third Workshop on Very Large Corpora",
    year = "1995",
    url = "https://www.aclweb.org/anthology/W95-0107",
}
```
