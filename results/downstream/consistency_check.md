# Downstream mixed models following scripts_renew.R
- N subjects: 49
- Rows: 19571

## Task descriptives
- single: accept=0.592, RT=941.4 ms
- double: accept=0.518, RT=867.4 ms

## Cooperation-task utility models (lme4)
- Choice (lme4::glmer): loss=4.003 (p=2.2e-09); defect=8.662 (p=9.23e-13); contrast p=0.000153
- RT (lme4::lmer): loss=25.366 (p=0.000562); defect=-35.347 (p=0.04); contrast p=0.002192
- P3 (lme4::lmer): loss=-0.074 (p=0.566); defect=0.450 (p=0.0146); contrast p=0.05241
- LPP (lme4::lmer): loss=-0.326 (p=0.0163); defect=0.681 (p=0.000413); contrast p=0.0001733


## Affective ratings
- N subjects: 42
- Outcome F=297.006, p=2.181e-20
- Task F=0.598, p=0.4438
- Outcome x task F=16.747, p=0.0001952
