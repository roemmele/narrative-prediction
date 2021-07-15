# narrative-prediction
This repository contains Python code that performs "narrative continuation" (i.e. predicting what happens next in a story) in a few different frameworks. This code is not actively maintained so it might not work with current versions of the library dependencies. I recommend creating a conda environment and installing the dependencies given in requirements.txt, e.g.:

```
conda create --name narrative_pred_env --file requirements.txt
conda activate narrative_pred_env
python -m spacy download en_core_web_md
```

See each folder for a detailed description of each framework with instructions for running the code:

## lm-generation: 
Code that generates new sentences in stories

## ROC:
Code that performs the Story Cloze Test, which predicts endings for stories

## COPA:
Code that performs the Choice of Plausible Alternatives (COPA), which predicts sentences that describe a cause-effect relation between story events
