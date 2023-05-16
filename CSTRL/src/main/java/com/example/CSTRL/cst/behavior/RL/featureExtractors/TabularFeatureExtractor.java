package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;

public abstract class TabularFeatureExtractor extends FeatureExtractor {
    public String getIdentifier(ArrayList<Double> S) {
        ArrayList<Double> features = extractFeatures(S);
        final int featureSize = features.size();

        StringBuilder id = new StringBuilder();
        if (featureSize >= 1) {
            id.append(features.get(0).toString());

            for (int i = 1; i < featureSize; i++) {
                id.append("-").append(features.get(i).toString());
            }
        }

        return id.toString();
    }

    public String getIdentifier(ArrayList<Double> S, ArrayList<Double> A) {
        ArrayList<Double> newS = new ArrayList<Double>();
        newS.addAll(S);
        newS.addAll(A);

        return getIdentifier(newS);
    }
}
