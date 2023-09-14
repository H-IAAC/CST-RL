package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;

public class LFAFroggerFeatureExtractor extends FeatureExtractor {
    // We'll try
    //
    // [action, raycast * action] for each action for each raycast

    int totalRaycasts;
    int totalActions;

    public LFAFroggerFeatureExtractor(int totalRaycasts, int totalActions) {
        this.totalRaycasts = totalRaycasts;
        this.totalActions = totalActions;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        S = normalizeValues(S);
        ArrayList<Double> features = new ArrayList<>();

        for (int i = 2 + totalRaycasts; i < 2 + totalRaycasts + totalActions; i++) {
            features.add(S.get(i));
            for (int j = 2; j < 2 + totalRaycasts; j++) {
                features.add(S.get(i) * S.get(j));
            }
        }

        return features;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        return totalActions + totalActions * totalRaycasts;
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        return null;
    }
}
