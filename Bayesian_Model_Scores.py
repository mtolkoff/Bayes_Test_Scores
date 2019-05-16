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

    reading_transform = logit(student_data['reading score'] / 101)

    with pm.Model() as hierarchical_model:
        intercept = pm.Normal('intercept', 0, sd=100)
        lunch_reading_beta = pm.Normal('lunch_reading_beta', 0, sd=100)
        gender_reading_beta = pm.Normal('gender_reading_beta', 0, sd=100)
        testPrep_reading_beta = pm.Normal('testPrep_reading_beta', 0, sd=100)
        race_reading_beta = pm.Normal('race_reading_beta', 0, sd=100, shape=(race_OneHot.shape[1]))
        PEd_reading_beta = pm.Normal('PEd_reading_beta', 0, sd=100, shape=(PEd_OneHot.shape[1]))

        eps = pm.HalfCauchy('eps', 5)

        students_lm = intercept + lunch_reading_beta * lunchOneHot + gender_reading_beta * genderOneHot + \
                      testPrep_reading_beta * testPrepOneHot
        for i in range(race_OneHot.shape[1]):
            students_lm = students_lm + race_reading_beta[i] * race_OneHot.iloc[:, i]
        for i in range(PEd_OneHot.shape[1]):
            students_lm = students_lm + PEd_reading_beta[i] * PEd_OneHot.iloc[:, i]

        reading = pm.Normal('reading', mu=students_lm, sd=eps, observed=reading_transform)

    with hierarchical_model:
        hm_trace = pm.sample(5000, tune=2000)
    pm.traceplot(hm_trace)
    plt.show()
