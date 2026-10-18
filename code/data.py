"""Dataset loaders. Each returns (X_df, y, s_crisp, mu, z, groups_for_split, meta).
  y        : binary label (np.int)
  s_crisp  : crisp protected indicator (1 = protected group), used by crisp baselines
  mu       : fuzzy membership of protected group in [0,1] (equals s_crisp for categorical attributes)
  z        : raw sensitive variable (for threshold-stability analysis) or None
  groups   : grouping variable for GroupShuffleSplit (Diabetes patients) or None
The protected attribute itself is excluded from model inputs (except where noted)."""
import numpy as np, pandas as pd, os
RAW = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def young(age, a=25.0, b=35.0):
    """Fuzzy set 'young': 1 for age<=a, linear to 0 at age>=b."""
    return np.clip((b - np.asarray(age, float)) / (b - a), 0.0, 1.0)

def adult():
    cols = "age workclass fnlwgt education education_num marital occupation relationship race sex cg cl hpw country income".split()
    kw = dict(header=None, names=cols, skipinitialspace=True, na_values='?')
    d = pd.concat([pd.read_csv(f'{RAW}/adult/adult.data', **kw),
                   pd.read_csv(f'{RAW}/adult/adult.test', skiprows=1, **kw)], ignore_index=True).dropna().reset_index(drop=True)
    y = d.income.str.contains('>').astype(int).values
    s = (d.sex == 'Female').astype(int).values
    X = d.drop(columns=['income', 'sex', 'education'])
    return X, y, s, s.astype(float), d.age.values, None, dict(name='Adult', attr='sex (female)', fuzzy=False)

def german():
    d = pd.read_csv(f'{RAW}/statloggermancreditdata/german.data', sep=' ', header=None)
    d.columns = [f'A{i+1}' for i in range(21)]
    y = (d.A21 == 1).astype(int).values           # 1 = good credit (favourable outcome)
    age = d.A13.values
    X = d.drop(columns=['A21', 'A13'])
    return X, y, (age <= 25).astype(int), young(age), age, None, dict(name='German', attr='age (young)', fuzzy=True)

def bank():
    d = pd.read_csv(f'{RAW}/bankmarketing/bank/bank-full.csv', sep=';')
    y = (d.y == 'yes').astype(int).values
    age = d.age.values
    X = d.drop(columns=['y', 'age', 'duration'])   # duration is post-outcome leakage (UCI documentation)
    return X, y, (age <= 25).astype(int), young(age), age, None, dict(name='Bank', attr='age (young)', fuzzy=True)

def default():
    d = pd.read_excel(f'{RAW}/defaultofcreditcardclients/default of credit card clients.xls', header=1)
    y = (d['default payment next month'] == 0).astype(int).values   # 1 = no default (favourable)
    s = (d.SEX == 2).astype(int).values
    X = d.drop(columns=['ID', 'default payment next month', 'SEX'])
    for c in ['EDUCATION', 'MARRIAGE']: X[c] = X[c].astype(str)
    return X, y, s, s.astype(float), d.AGE.values, None, dict(name='Default', attr='sex (female)', fuzzy=False)

def student():
    d = pd.read_csv(f'{RAW}/studentperformance/student/student-por.csv', sep=';')
    y = (d.G3 >= 10).astype(int).values
    s = (d.sex == 'M').astype(int).values          # lower pass rate group
    X = d.drop(columns=['G1', 'G2', 'G3', 'sex'])  # G1/G2: near-leakage (corr(G2,G3)=0.92)
    return X, y, s, s.astype(float), None, None, dict(name='Student', attr='sex (male)', fuzzy=False)

def compas():
    c = pd.read_csv(f'{RAW}/compas.csv')
    c = c[(c.days_b_screening_arrest.between(-30, 30)) & (c.is_recid != -1) & (c.c_charge_degree != 'O') & (c.score_text != 'N/A')].reset_index(drop=True)
    y = (c.two_year_recid == 0).astype(int).values  # 1 = no recidivism (favourable)
    s = (c.race == 'African-American').astype(int).values
    X = c[['age', 'sex', 'priors_count', 'c_charge_degree', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']].copy()
    return X, y, s, s.astype(float), c.age.values, None, dict(name='COMPAS', attr='race (African-American)', fuzzy=False)

def diabetes():
    d = pd.read_csv(f'{RAW}/diabetes130-ushospitalsforyears1999-2008/diabetic_data.csv', na_values='?', low_memory=False)
    d = d[d.gender != 'Unknown/Invalid'].reset_index(drop=True)
    y = (d.readmitted != '<30').astype(int).values  # 1 = not readmitted within 30 days (favourable)
    s = (d.race == 'AfricanAmerican').astype(int).values
    for c in ['diag_1', 'diag_2', 'diag_3']:
        v = d[c].astype(str).str[:3]; top = v.value_counts().index[:20]
        d[c] = np.where(v.isin(top), v, 'other')
    d['medical_specialty'] = d.medical_specialty.fillna('NA')
    top = d.medical_specialty.value_counts().index[:15]
    d['medical_specialty'] = np.where(d.medical_specialty.isin(top), d.medical_specialty, 'other')
    for c in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']: d[c] = d[c].astype(str)
    X = d.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'readmitted', 'race'])
    return X, y, s, s.astype(float), None, d.patient_nbr.values, dict(name='Diabetes', attr='race (African-American)', fuzzy=False)

def census_kdd():
    names = ['age','class_worker','industry_code','occupation_code','education','wage_per_hour','enroll_edu','marital','major_industry',
             'major_occupation','race','hispanic','sex','union','unemp_reason','ft_pt','cap_gain','cap_loss','dividends','tax_filer',
             'region_prev','state_prev','household_detail','household_summary','instance_weight','migr_msa','migr_reg','migr_within_reg',
             'same_house','migr_sunbelt','num_employer','family_u18','country_father','country_mother','country_self','citizenship',
             'self_employed','veterans_admin','veterans_benefits','weeks_worked','year','income']
    kw = dict(header=None, names=names, skipinitialspace=True, low_memory=False)
    d = pd.concat([pd.read_csv(f'{RAW}/censusincomekdd/census-income.data', **kw),
                   pd.read_csv(f'{RAW}/censusincomekdd/census-income.test', **kw)], ignore_index=True)
    y = d.income.astype(str).str.contains(r'50000\+').astype(int).values
    s = (d.sex == 'Female').astype(int).values
    for c in ['industry_code', 'occupation_code', 'own_business_or_self_employed' if False else 'self_employed', 'veterans_benefits', 'year']:
        d[c] = d[c].astype(str)
    X = d.drop(columns=['income', 'sex', 'instance_weight', 'household_detail'])  # instance_weight is a survey weight, not a feature
    return X, y, s, s.astype(float), d.age.values, None, dict(name='CensusKDD', attr='sex (female)', fuzzy=False)

LOADERS = dict(German=german, Student=student, COMPAS=compas, Default=default, Bank=bank, Adult=adult, Diabetes=diabetes, CensusKDD=census_kdd)

def split_and_encode(X, y, s, mu, groups, seed, max_train=None):
    """60/20/20 split stratified on (y, s) (or grouped by patient); one-hot + standardise fit on train only."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    n = len(y); idx = np.arange(n)
    if groups is not None:
        g1 = GroupShuffleSplit(1, test_size=0.2, random_state=seed); tr_va, te = next(g1.split(idx, groups=groups))
        g2 = GroupShuffleSplit(1, test_size=0.25, random_state=seed); a, b = next(g2.split(tr_va, groups=groups[tr_va]))
        tr, va = tr_va[a], tr_va[b]
    else:
        strat = y * 2 + s
        tr_va, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
        tr, va = train_test_split(tr_va, test_size=0.25, random_state=seed, stratify=strat[tr_va])
    if max_train and len(tr) > max_train:
        rng = np.random.default_rng(seed); tr = rng.choice(tr, max_train, replace=False)
        va = rng.choice(va, min(len(va), max_train // 3), replace=False); te = rng.choice(te, min(len(te), max_train // 3), replace=False)
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith(('str', 'category'))]
    num = [c for c in X.columns if c not in cat]
    Xd = pd.get_dummies(X, columns=cat, dtype=float)
    keep = Xd.columns[(Xd.iloc[tr].std() > 0)]
    Xd = Xd[keep].astype(float)
    mean, std = Xd.iloc[tr][num].mean(), Xd.iloc[tr][num].std()
    Xd[num] = (Xd[num] - mean) / std.replace(0, 1)
    Xd[num] = Xd[num].clip(-5, 5)
    A = Xd.values.astype(np.float32)
    out = {k: (A[i], y[i], s[i], mu[i], i) for k, i in [('tr', tr), ('va', va), ('te', te)]}
    return out, list(Xd.columns)

if __name__ == '__main__':
    for k, f in LOADERS.items():
        X, y, s, mu, z, g, m = f()
        sp, cols = split_and_encode(X, y, s, mu, g, 0)
        print(f"{k:10s} n={len(y):6d} d_enc={len(cols):4d} pos={y.mean():.3f} prot={s.mean():.3f} E[mu]={mu.mean():.3f} "
              f"P(y|prot)={y[s==1].mean():.3f} P(y|rest)={y[s==0].mean():.3f}  tr/va/te={[len(sp[t][1]) for t in ('tr','va','te')]}")
