## Permutation Importance Physics

This package helps calculate permutation importance of features for a given model based on a given metric to measure performance. The idea is to assess the performance of a _trained model_ based on a given performance metric if one input feature were taking away (values of the feature are shuffled between samples).

Results are usually more meaningful than _feature importance_ provided by Boosted Decision Trees, provide an uncertainty measure, and allow more flexibility to choose evaluation metric as well as test dataset. It is faster to compute compared to _iterative removal_ method because there is no re-training.

This package handles 'sample_weights', custom performance metrics, and provides some predefined High Energy Physics based metrics.  It can be used to evaluate feature importance on a given dataset that is not necessarily from the same distribution as the training set, so the feature importances of a trained model can be recalculated for a new test dataset, and/or evaluation metric. This might be useful to test sensitivity to systematic shifts (domain adaptation), or just the impact of features for particular subsets of the dataset (samples with 1 jet, 2 jets, samples with a score > 0.6, signal at mass 700 GeV, 800 GeV, etc).

>WARNING: Choosing the right metric is essential to get meaningful results. Make sure to check if the value of PI for your given features makes sense. If 'discovery significance' is your metric (which usually ranges between 0 and 6), a permutation importance of 112 for a particular feature should worry you.

When in doubt, use 'AUC' as a reasonable metric for a classification problem, rather than 'accuracy'.


>WARNING: With random forrests or DNN with dropouts, the PI for 2 correlated features might be 0 because dropping any one individually does not hamper the performance of the model, however dropping both might decrease performance. In this package the PI is calculated by dropping only 1 feature at a time for now. Future versions might provide an option to calculate PI taking into account correlations, if there is demonstrated interest.

### Quick tutorial
    pip install PermutationImportancePhysics

In `Python3`

    from permutationimportancephysics.PermutationImportance import PermulationImportance
    pi = PermulationImportance(model=bdt, X=X_test,y=y_test,weights=weights_test,n_iterations=3,usePredict_poba=True,
                          scoreFunction="AUC")
    pi.dislayResults()

Or for discovery significance

    pi = PermulationImportance(model=bdt, X=X_test,y=y_test,weights=weights_test,n_iterations=3,usePredict_poba=True,
                           scoreFunction="amsasimov")
    pi.dislayResults()

Plot feature importances with error bars

    plt = pi.plotBars()
    plt.show()

`n_iterations(default=3)`:  number of times the permutation importance of a feature is calculated after a new shuffle. Higher => smaller uncertainty, more computation time.
`usePredict_poba(default=False)`: use `model.predict_proba()` instead of `model.predict()`, useful for SKLearn models.
`scoreFunction(default='AUC')`: evaluation metric used to calculate permutation importance over the entire evaluation dataset. User defined function possible of the form: `func (X_eval, y_true, weights)`.

### Dependencies:
- numpy
- matplotlib
- sklearn

### ToDo:

- Multiprocessing
- AUC with negative weight handling