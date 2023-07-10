package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;

public class LFAFroggerFeatureExtractor extends FeatureExtractor {
    int totalRaycasts;
    int totalActions;

    public LFAFroggerFeatureExtractor(int totalRaycasts, int totalActions) {
        this.totalRaycasts = totalRaycasts;
        this.totalActions = totalActions;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        S = normalizeValues(S);

        ArrayList<Double> features = new ArrayList<>(S);

        for (int i = 3 + totalRaycasts; i < 3 + totalRaycasts + totalActions; i++) {
            features.add(S.get(2) * S.get(i));
        }

        for (int i = 3; i < 3 + totalRaycasts; i++) {
            for (int j = 3 + totalRaycasts; j < 3 + totalRaycasts + totalActions; j++) {
                features.add(S.get(i) * S.get(j));
                for (int k = 0; k < 3; k++) {
                    features.add(S.get(i) * S.get(j) * S.get(k));
                }
            }
        }

        return features;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        return 3 + totalRaycasts + totalActions + totalActions + 4 * totalRaycasts * totalActions;
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        return null;
    }
}
