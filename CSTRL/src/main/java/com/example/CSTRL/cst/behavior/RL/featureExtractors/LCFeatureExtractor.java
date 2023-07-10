package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;
import java.util.List;

public class LCFeatureExtractor extends FeatureExtractor {

    int power;

    public LCFeatureExtractor(int power) {
        this.power = power;
    }

    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        S = normalizeValues(S);

        ArrayList<Double> features = new ArrayList<Double>(List.of(1.0));

        for (int i = 0; i < S.size(); i++) {
            for (int j = 1; j < power + 1; j++) {
                for (int k = 0; k < Math.pow(power + 1, i); k++) {
                    features.add(Math.pow(S.get(i), j) * features.get(k));
                }
            }
        }

        return features;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        return (int) Math.pow(power + 1, stateSize);
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        return null;
    }
}
