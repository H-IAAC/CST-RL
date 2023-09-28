package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import java.util.ArrayList;

public class DirectFeatureExtractor extends FeatureExtractor {
    @Override
    public ArrayList<Double> extractFeatures(ArrayList<Double> S) {
        return S;
    }

    @Override
    public int getFeatureVectorSize(int stateSize) {
        return stateSize;
    }

    @Override
    public ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A) {
        return null;
    }

    @Override
    public String toString() {
        return "DirectFE";
    }
}
