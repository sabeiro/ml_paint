import numpy as np
import os
import text_gen as t_g
import importlib
import random

baseDir = "/home/sabeiro/tmp/pers/"
cName = "markdown"
cName = "anima"
#content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
opt = {"sequence_length":100,"batch_size":128,"n_epoch":30,"baseDir":baseDir+"/text/","cName":cName,"fName":baseDir+"text/"+cName+".txt","isLoad":True,"genCoding":False}

text = open(opt['fName'],encoding="utf-8").read()
#opt['isLoad'] = False
#gen.gen_coding(text)
#gen.load_vocab(opt['baseDir'] + "english_vocab.txt")

importlib.reload(t_g)
gen = t_g.text_gen(opt)
gen = t_g.text_gen(opt)

gen.train(n_epoch=10,text=text)
gen.save_model()

text = gen.clean_text(text)
text = re.sub("\n"," ",text)
for i in range(10):
    n = int(random.uniform(0,len(text)))
    seed = text[n:n+100]
    generated = gen.gen(seed=seed,n_chars=150)
    print(seed + ' -> ' + generated)


