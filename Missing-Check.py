import pandas as pd
import pymc3 as pm
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy

if __name__ == '__main__':
    def logit(p):
        return np.log(p) - np.log(1 - p)


    student_data = pd.read_csv('StudentsPerformance.csv')
    #student_data = student_data.sample(200)

    id0 = []
    id1 = []
    for i in range(100):
        randid0 = random.randint(0, student_data.shape[0]-1)
        randid1 = random.randint(0, 4)
        id0.append(randid0)
        id1.append(randid1)
        student_data.iloc[randid0, randid1] = np.nan

    f = open('missing_indices.txt', 'w')
    for i in range(len(id0)):
        f.write(str(id0[i]) + ', ' + str(id1[i]) + "\n")
    f.close()

    lunchOneHot = pd.get_dummies(student_data['lunch'], dummy_na=True, drop_first=True)
    genderOneHot = pd.get_dummies(student_data['gender'], dummy_na=True, drop_first=True)
    testPrepOneHot = pd.get_dummies(student_data['test preparation course'], dummy_na=True, drop_first=True)

    race_OneHot = pd.get_dummies(student_data['race/ethnicity'], dummy_na=True, drop_first=True)
    PEd_OneHot = pd.get_dummies(student_data['parental level of education'], dummy_na=True, drop_first=True)

    reading_transform = logit((student_data['reading score'] + 1) / 102)
    writing_transform = logit((student_data['writing score'] + 1) / 102)
    math_transform = logit((student_data['math score'] + 1) / 102)

    genderOneHot.loc[np.squeeze(genderOneHot.iloc[:, 1:2] == 1)] = np.nan
    genderOneHot = genderOneHot.iloc[:, 0:1]

    lunchOneHot.loc[np.squeeze(lunchOneHot.iloc[:, 1:2] == 1)] = np.nan
    lunchOneHot = lunchOneHot.iloc[:, 0:1]

    testPrepOneHot.loc[np.squeeze(testPrepOneHot.iloc[:, 1:2] == 1)] = np.nan
    testPrepOneHot = testPrepOneHot.iloc[:, 0:1]

    race_OneHot.loc[np.squeeze(race_OneHot.iloc[:, 4:5] == 1)] = np.nan
    race_OneHot = race_OneHot.iloc[:, 0:4]

    PEd_OneHot.loc[np.squeeze(PEd_OneHot.iloc[:, 5:6] == 1)] = np.nan
    PEd_OneHot = PEd_OneHot.iloc[:, 0:5]
    print(np.isnan(PEd_OneHot).sum())

    with pm.Model() as hierarchical_model:
        lunch_score = pm.Normal('lunch_score', 0, 1, observed=lunchOneHot)
        test_prep_score = pm.Normal('test_prep_score', 0, 1, observed=testPrepOneHot)
        gender_score = pm.Normal('gender_score', 0, 1, observed=genderOneHot)
        PEd_score = pm.Normal('PEd_score', 0, 1, observed=PEd_OneHot, shape=PEd_OneHot.shape)
        race_score = pm.Normal('race_score', 0, 1, observed=race_OneHot, shape=race_OneHot.shape)

        # Reading
        intercept_mean = pm.Normal('intercept_mean', 0, sd=100)
        lunch_mean = pm.Normal('lunch_mean', 0, sd=100)
        gender_mean = pm.Normal('gender_mean', 0, sd=100)
        testPrep_mean = pm.Normal('testPrep_mean', 0, sd=100)
        race_mean = pm.Normal('race_mean', 0, sd=100, shape=race_OneHot.shape[1])
        PEd_mean = pm.Normal('PEd_mean', 0, sd=100, shape=PEd_OneHot.shape[1])

        global_shrinkage_model = pm.InverseGamma('global_shrinkage_model', mu=1, sd=100)
        local_shrinkage_model = pm.InverseGamma('local_shrinkage_model', mu=1, sd=100, shape=3)

        global_shrinkage_beta = pm.InverseGamma('global_shrinkage_beta', mu=1, sd=100)
        local_shrinkage_beta = pm.InverseGamma('local_shrinkage_beta', mu=1, sd=100, shape=39)

        intercept_reading = pm.Normal('intercept_reading', intercept_mean,
                                      sd=global_shrinkage_beta * local_shrinkage_beta[0])
        lunch_reading_beta = pm.Normal('lunch_reading_beta', lunch_mean,
                                       sd=global_shrinkage_beta * local_shrinkage_beta[1])
        gender_reading_beta = pm.Normal('gender_reading_beta', gender_mean,
                                        sd=global_shrinkage_beta * local_shrinkage_beta[2])
        testPrep_reading_beta = pm.Normal('testPrep_reading_beta', testPrep_mean,
                                          sd=global_shrinkage_beta * local_shrinkage_beta[3])
        race_reading_beta = pm.Normal('race_reading_beta', race_mean,
                                      sd=global_shrinkage_beta * local_shrinkage_beta[4:8],
                                      shape=(race_OneHot.shape[1]))
        PEd_reading_beta = pm.Normal('PEd_reading_beta', PEd_mean,
                                     sd=global_shrinkage_beta * local_shrinkage_beta[8:13], shape=(PEd_OneHot.shape[1]))

        students_lm_reading = intercept_reading + lunch_reading_beta * lunch_score + gender_reading_beta * gender_score + \
                              testPrep_reading_beta * test_prep_score
        students_lm_reading = students_lm_reading + pm.math.dot(race_score, race_reading_beta)
        students_lm_reading = students_lm_reading + pm.math.dot(PEd_score, PEd_reading_beta)

        reading_score = pm.Normal('reading', mu=students_lm_reading,
                                  sd=local_shrinkage_model[0], observed=reading_transform)

        # Writing
        intercept_writing = pm.Normal('intercept_writing', intercept_mean,
                                      sd=global_shrinkage_beta * local_shrinkage_beta[13])
        lunch_writing_beta = pm.Normal('lunch_writing_beta', lunch_mean,
                                       sd=global_shrinkage_beta * local_shrinkage_beta[14])
        gender_writing_beta = pm.Normal('gender_writing_beta', gender_mean,
                                        sd=global_shrinkage_beta * local_shrinkage_beta[15])
        testPrep_writing_beta = pm.Normal('testPrep_writing_beta', testPrep_mean,
                                          sd=global_shrinkage_beta * local_shrinkage_beta[16])
        race_writing_beta = pm.Normal('race_writing_beta', race_mean,
                                      sd=global_shrinkage_beta * local_shrinkage_beta[17:21],
                                      shape=(race_OneHot.shape[1]))
        PEd_writing_beta = pm.Normal('PEd_writing_beta', PEd_mean, sd=global_shrinkage_beta * local_shrinkage_beta[21:26
                                                                                              ],
                                     shape=(PEd_OneHot.shape[1]))

        students_lm_writing = intercept_writing + lunch_writing_beta * lunch_score + gender_writing_beta * gender_score \
                              + testPrep_writing_beta * test_prep_score
        students_lm_writing = students_lm_writing + pm.math.dot(race_score, race_writing_beta)
        students_lm_writing = students_lm_writing + pm.math.dot(PEd_score, PEd_writing_beta)

        writing_score = pm.Normal('writing', mu=students_lm_writing, sd=local_shrinkage_model[1]
                                  , observed=writing_transform)

        intercept_math = pm.Normal('intercept_math', intercept_mean, sd=global_shrinkage_beta * local_shrinkage_beta[26]
                                   )
        lunch_math_beta = pm.Normal('lunch_math_beta', lunch_mean, sd=global_shrinkage_beta * local_shrinkage_beta[27])
        gender_math_beta = pm.Normal('gender_math_beta', gender_mean, sd=global_shrinkage_beta * local_shrinkage_beta[28
        ])
        testPrep_math_beta = pm.Normal('testPrep_math_beta', testPrep_mean, sd=global_shrinkage_beta *
                                                                               local_shrinkage_beta[29])
        race_math_beta = pm.Normal('race_math_beta', race_mean, sd=global_shrinkage_beta * local_shrinkage_beta[30:34],
                                   shape=(race_OneHot.shape[1]))
        PEd_math_beta = pm.Normal('PEd_math_beta', PEd_mean, sd=global_shrinkage_beta * local_shrinkage_beta[34:39],
                                  shape=(PEd_OneHot.shape[1]))

        students_lm_math = intercept_math + lunch_math_beta * lunch_score + gender_math_beta * gender_score + \
                           testPrep_math_beta * test_prep_score
        students_lm_math = students_lm_math + pm.math.dot(race_score, race_math_beta)
        students_lm_math = students_lm_math + pm.math.dot(PEd_score, PEd_math_beta)

        math_score = pm.Normal('math_score', mu=students_lm_math, sd=local_shrinkage_model[2],
                               observed=math_transform)



    with hierarchical_model:
        # step = pm.Metropolis([PEd_math_beta, race_math_beta, testPrep_math_beta, gender_math_beta, lunch_math_beta,
        # intercept_math, PEd_writing_beta, race_writing_beta, testPrep_writing_beta,
        # gender_writing_beta, lunch_writing_beta, intercept_writing, PEd_reading_beta,
        # race_reading_beta, testPrep_reading_beta, gender_reading_beta, lunch_reading_beta,
        # intercept_reading, local_shrinkage_beta, global_shrinkage_beta,
        # local_shrinkage_model, global_shrinkage_model, PEd_mean, race_mean, testPrep_mean,
        # gender_mean, lunch_mean, intercept_mean])
        # pm.sampling.init_nuts(init='adapt_diag')
        db = pm.backends.Text('missingoutput.csv')
        # hm_trace = pm.sample(100 * 100, step=step, trace=db)
        hm_trace = pm.sample(1000, tune=1000,
                             init='advi+adapt_diag',
                             trace=db,
                             njobs=1
                             )
