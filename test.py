# -*- coding: utf-8 -*-

comments = ["i know", "i think that will rain"]


toxic = [0, 0]
severe_toxic = [0, 0]
obscene = [0, 0]
threat = [0, 0]
insult = [0, 0]
identity_hate = [0, 0]

data = {"comment_text": comments, "toxic": toxic, "severe_toxic": severe_toxic,
        "obscene": obscene, "threat": threat, "insult": insult, "identity_hate": identity_hate}

from modules.pre_process import PreProcessStage
import pandas as pd
import re

df = pd.DataFrame(data=data)
print(df)
print(re.sub("[^\w]", " ", df["comment_text"][0]).split())

preprocess = PreProcessStage(df)
preprocess.pre_process_dataset()
print(preprocess.word_2_idx)
print(preprocess.input_data)
print()
