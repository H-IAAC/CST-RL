package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import com.example.CSTRL.cst.behavior.RL.RLElement;

import java.util.ArrayList;

public abstract class FeatureExtractor extends RLElement {
    public abstract ArrayList<Double> extractFeatures(ArrayList<Double> S);

    public ArrayList<Double> extractFeatures(ArrayList<Double> S, ArrayList<Double> A) {
        ArrayList<Double> newS = new ArrayList<Double>();
        newS.addAll(S);
        newS.addAll(A);

        return extractFeatures(newS);
    }

    public abstract int getFeatureVectorSize(int stateSize);
}
