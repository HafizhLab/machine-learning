# Language Model Prediction Model for Quranic Corpus
Prediction by word is implemented using N-gram Language Model approach which is modified from Shah (2020) version.

# Example Usage
Train model to learn lookup dictionary for N-gram prediction. Model can only be trained on string input (``ayah: String``) using ``add_document`` function.
```python
from lm.markovchain import MarkovChain

# load model
model = MarkovChain()

# train
for ayah in ayahs:
    model.add_document(ayah)
```

Predict next word
```python
# test model input: وَاللَّهُ
model.next_word(input())
# output [('عَلِيمٌ', 24), ('لَا', 19), ('يَعْلَمُ', 16)]

# test model input: وَاللَّهُ بِمَا تَعْمَلُونَ
model.next_word(input())
# output [('خَبِيرٌ', 7), ('بَصِيرٌ', 6), ('عَلِيمٌ', 2)]
```

Pre-trained model is provided in ``lm_alquran.pickle`` and training corpus is provided in ``alquran_korpus.txt``. Corpus is obtained from [Al Quran Cloud](http://api.alquran.cloud/) using simple arabic edition.

# Reference
Shah, D. (2020, May 10). Exploring the Next Word Predictor! Retrieved January 24, 2021, from https://towardsdatascience.com/exploring-the-next-word-predictor-5e22aeb85d8f
