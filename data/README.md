# Data

The raw datasets are **not** redistributed here. Download them from their original sources and unpack them
so that `data/raw/` has exactly this layout (the loaders in `code/data.py` expect these paths):

```
data/raw/
  adult/adult.data, adult/adult.test                                   UCI id 2   https://archive.ics.uci.edu/dataset/2/adult
  statloggermancreditdata/german.data                                  UCI id 144 https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
  bankmarketing/bank/bank-full.csv                                     UCI id 222 https://archive.ics.uci.edu/dataset/222/bank+marketing
  defaultofcreditcardclients/default of credit card clients.xls        UCI id 350 https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
  studentperformance/student/student-por.csv                           UCI id 320 https://archive.ics.uci.edu/dataset/320/student+performance
  diabetes130-ushospitalsforyears1999-2008/diabetic_data.csv           UCI id 296 https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
  censusincomekdd/census-income.data, census-income.test               UCI id 117 https://archive.ics.uci.edu/dataset/117/census+income+kdd
  compas.csv                                                           ProPublica
```

Notes:
- The Bank Marketing and Student Performance downloads contain nested zip files; unzip them in place.
- Census-Income (KDD) is distributed as `census.tar.gz`; extract it into `censusincomekdd/`.
- COMPAS: `curl -o data/raw/compas.csv https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv`

Check the licence of each source before redistributing any file.
