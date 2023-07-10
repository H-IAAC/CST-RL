package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;
import java.util.Collections;

public class SimplifiedContinuousLFAFeatureExtractor extends FeatureExtractor {
    // Here, as well as a_i for each action, for each combination of (a_i, p_i), a_i * p_i is in our feature vector and (a_i^2, p_i)
    // NOTE: Doesn't seem to be converging

    final int totalRaycasts;
    final int totalActions;

    public SimplifiedContinuousLFAFeatureExtractor(int totalRaycasts, int totalActions) {
        this.totalRaycasts = totalRaycasts;
        this.totalActions = totalActions;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        S = normalizeValues(S);

        ArrayList<Double> features = new ArrayList<>();

        for (int i = 0; i < totalActions; i++) {
            features.add(S.get(3 + totalRaycasts + i));
        }

        for (int i = 0; i < totalActions; i ++) {
            for (int j = 0; j < totalRaycasts; j++) {
                features.add(S.get(3 + totalRaycasts + i) * S.get(3 + j));
                features.add(Math.pow(S.get(3 + totalRaycasts + i), 2) * S.get(3 + j));
            }
        }

        return features;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        return totalActions + 2 * totalActions * totalRaycasts;
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        ArrayList<Double> combinedS = new ArrayList<>(S);
        combinedS.addAll(A);

        ArrayList<Double> normalizedS = normalizeValues(combinedS);

        ArrayList<ArrayList<Double>> actionJacobian = new ArrayList<>();
        for (int i = 0; i < totalActions; i++) {
            actionJacobian.add(new ArrayList<>(Collections.nCopies(getFeatureVectorSize(normalizedS.size()), 0.0)));
        }

        for (int i = 0; i < totalActions; i ++) {
            actionJacobian.get(i).set(i, 1.0);
            for (int j = 0; j < totalRaycasts; j++) {
                actionJacobian.get(i).set(totalActions + i * totalRaycasts + 2 * j, normalizedS.get(3 + j));
                actionJacobian.get(i).set(totalActions + i * totalRaycasts + 2 * j + 1, 2 * normalizedS.get(3 + totalRaycasts + i) * normalizedS.get(3 + j));
            }
        }
        return actionJacobian;
    }
}
