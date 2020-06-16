# Neural UMLS Concept Linker
This repository contains an implementation of a contextualized neural concept linking system for clinical notes.

Given a clinical note and span (mention) annotations over the note, this system normalizes each span to it's corresponding UMLS Concept with high accuracy.



# Use
To run this project locally, first clone it:

```bash
git clone https://github.com/elliotschu/concept-linker.git
```

and add the repository to your python path.



Then enter and modify the `config.ini` file. Relevant configurations you may want to change are as follows:

- `root`: a directory to cache results during both training and prediction.
- `lexicon`: a TSV file containing the concept string, concept id (CUI) on each line. This should be initialized to reflect the ontology or UMLS version one is training/predicting against.


# Data
During training and prediction, annotations are expected at the document level. Specifically, list of dictionaries as follows is expected:

    clinical note id                   : document unique identifier (this is could be a file name)
    clinical note                      : (spacy Doc object) tokenized and sentence segmented by language.py
    list of mention annotations dicts  : dictionaries containg a file relative 'index' and an array of 'mention' Span objects
    {
        'id': ...
        'note': Doc('note text')
        'mentions':[]
    }

    where

    mentions[i] = {
        'concept': umls cui (or None if for prediction)
        'concept_name': cui name
        'mention': [] (an array spacy Span objects belonging to 'note' that contain the mention annotation spans.)
    }

    mention: a list of spacy Span objects containing the annotated concept mentions with attributes ._.cui
              this attribute is set to None during prediction to indicate that the span requires an annotation to be set.


In the file `codebase/linker.py`, `load_data` shows an example of such data being utilized.
