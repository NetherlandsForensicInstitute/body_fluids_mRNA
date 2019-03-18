from sklearn.neural_network import MLPClassifier

from reimplementation import *

class ScoresMLP():

    def __init__(self, random_state=0):
        self._mlp = MLPClassifier(random_state=random_state)

    def fit(self, X, y):
        """
        Train MLP model.
        """
        self._mlp.fit(X, y)
        return self


    def predict_proba_per_class(self, Xtest, y_n_hot, labels_in_class, classes_map, MAX_LR):
        """
        Predicts probabilties per class.
        """
        ypred_proba = self._mlp.predict_proba(Xtest)

        h1_h2_probs_per_class = {}
        # marginal for each single class sample
        prob_per_class = convert_prob_per_mixture_to_marginal_per_class(
            ypred_proba, labels_in_class, classes_map, MAX_LR)
        for idx, celltype in enumerate(sorted((classes_map))):
            print(celltype)
            i_celltype = classes_map[celltype]
            # get the probability per single class sample
            total_proba = prob_per_class[:, idx]
            if sum(total_proba) > 0:
                probas_without_cell_type = total_proba[y_n_hot[:, i_celltype] == 0]
                probas_with_cell_type = total_proba[y_n_hot[:, i_celltype] == 1]
                h1_h2_probs_per_class[celltype] = (probas_with_cell_type, probas_without_cell_type)

        return h1_h2_probs_per_class
