import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy

if __name__ == '__main__':
    def logit(p):
        return np.log(p) - np.log(1 - p)

    student_data = pd.read_csv('StudentsPerformance.csv')

    lunchOneHot = pd.get_dummies(student_data['lunch']).iloc[:, 1:2]
    genderOneHot = pd.get_dummies(student_data['gender']).iloc[:, 1:2]
    testPrepOneHot = pd.get_dummies(student_data['test preparation course']).iloc[:, 1:2]

    race_OneHot = pd.get_dummies(student_data['race/ethnicity']).iloc[:, 1:5]
    PEd_OneHot = pd.get_dummies(student_data['parental level of education']).iloc[:, 1:6]

    reading_transform = logit((student_data['reading score'] + 1) / 102)
    writing_transform = logit((student_data['writing score'] + 1) / 102)
    math_transform = logit((student_data['math score'] + 1) / 102)

    with pm.Model() as hierarchical_model:
        #Reading
        intercept_reading = pm.Normal('intercept_reading', 0, sd=100)
        lunch_reading_beta = pm.Normal('lunch_reading_beta', 0, sd=100)
        gender_reading_beta = pm.Normal('gender_reading_beta', 0, sd=100)
        testPrep_reading_beta = pm.Normal('testPrep_reading_beta', 0, sd=100)
        race_reading_beta = pm.Normal('race_reading_beta', 0, sd=100, shape=(race_OneHot.shape[1]))
        PEd_reading_beta = pm.Normal('PEd_reading_beta', 0, sd=100, shape=(PEd_OneHot.shape[1]))

        eps = pm.HalfCauchy('eps', 5)

        students_lm_reading = intercept_reading + lunch_reading_beta * lunchOneHot + gender_reading_beta * genderOneHot + \
                      testPrep_reading_beta * testPrepOneHot
        for i in range(race_OneHot.shape[1]):
            students_lm_reading = students_lm_reading + race_reading_beta[i] * race_OneHot.iloc[:, i]
        for i in range(PEd_OneHot.shape[1]):
            students_lm_reading = students_lm_reading + PEd_reading_beta[i] * PEd_OneHot.iloc[:, i]

        reading_score = pm.Normal('reading', mu=students_lm_reading, sd=eps, observed=reading_transform)

        #Writing
        intercept_writing = pm.Normal('intercept_writing', 0, sd=100)
        lunch_writing_beta = pm.Normal('lunch_writing_beta', 0, sd=100)
        gender_writing_beta = pm.Normal('gender_writing_beta', 0, sd=100)
        testPrep_writing_beta = pm.Normal('testPrep_writing_beta', 0, sd=100)
        race_writing_beta = pm.Normal('race_writing_beta', 0, sd=100, shape=(race_OneHot.shape[1]))
        PEd_writing_beta = pm.Normal('PEd_writing_beta', 0, sd=100, shape=(PEd_OneHot.shape[1]))

        students_lm_writing = intercept_writing + lunch_writing_beta * lunchOneHot + gender_writing_beta * genderOneHot + \
                      testPrep_writing_beta * testPrepOneHot
        for i in range(race_OneHot.shape[1]):
            students_lm_writing = students_lm_writing + race_writing_beta[i] * race_OneHot.iloc[:, i]
        for i in range(PEd_OneHot.shape[1]):
            students_lm_writing = students_lm_writing + PEd_writing_beta[i] * PEd_OneHot.iloc[:, i]

        writing_score = pm.Normal('writing', mu=students_lm_writing, sd=eps, observed=writing_transform)

        intercept_math = pm.Normal('intercept_math', 0, sd=100)
        lunch_math_beta = pm.Normal('lunch_math_beta', 0, sd=100)
        gender_math_beta = pm.Normal('gender_math_beta', 0, sd=100)
        testPrep_math_beta = pm.Normal('testPrep_math_beta', 0, sd=100)
        race_math_beta = pm.Normal('race_math_beta', 0, sd=100, shape=(race_OneHot.shape[1]))
        PEd_math_beta = pm.Normal('PEd_math_beta', 0, sd=100, shape=(PEd_OneHot.shape[1]))

        students_lm_math = intercept_math + lunch_math_beta * lunchOneHot + gender_math_beta * genderOneHot + \
                      testPrep_math_beta * testPrepOneHot
        for i in range(race_OneHot.shape[1]):
            students_lm_math = students_lm_math + race_math_beta[i] * race_OneHot.iloc[:, i]
        for i in range(PEd_OneHot.shape[1]):
            students_lm_math = students_lm_math + PEd_math_beta[i] * PEd_OneHot.iloc[:, i]

        math_score = pm.Normal('math_score', mu=students_lm_math, sd=eps, observed=math_transform)


    with hierarchical_model:
        hm_trace = pm.sample(500, tune=200)
    pm.traceplot(hm_trace)
    plt.show()
