# data/ (gitignored)

Owner: CS teammate. NHANES source files cached here; never committed.

## What to download (1999–2008 continuous cycles + Linked Mortality File)

Per cycle, the component files you need (CDC NHANES, `.XPT`):
- **DEMO** — demographics (`RIDAGEYR` age, `RIAGENDR` sex, `RIDRETH1` race/eth, `SDDSRVYR` cycle, `SEQN`)
- **BPX** — blood pressure (`BPXSY1`…)
- **BMX** — body measures (`BMXBMI`)
- **TCHOL/LBXTC** — total cholesterol; **GHB** — HbA1c
- **SMQ** — smoking; **MCQ** — medical conditions (cancer, etc.); **DIQ** — diabetes; **RXQ** — meds

Cycles: `1999-2000, 2001-2002, 2003-2004, 2005-2006, 2007-2008` (≥10-yr follow-up before the 2019 LMF cutoff → 10-yr horizon fully observed).

**Linked Mortality File (public-use):** linked to NDI through **Dec 31 2019**. Fixed-width `.dat`; key fields: `SEQN`, `MORTSTAT` (0 alive / 1 deceased), `PERMTH_INT` (person-months from interview), `UCOD_LEADING`.

CDC pages:
- NHANES questionnaires/datasets: https://wwwn.cdc.gov/nchs/nhanes/
- Linked Mortality Files: https://www.cdc.gov/nchs/data-linkage/mortality-public.htm

`pyreadstat` or `pandas.read_sas(..., format="xport")` reads `.XPT`. The LMF layout doc gives the fixed-width column spec.

Drop everything here; `src/nhanes/build_cohort.py` joins on `SEQN` with censoring-correct outcomes.
