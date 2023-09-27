package com.example.CSTRL.cst.behavior.RL.featureExtractors;

import com.example.CSTRL.cst.behavior.RL.RLElement;

import java.util.ArrayList;

public abstract class FeatureExtractor {
    private ArrayList<Double> maxStateValues = new ArrayList<>();

    public abstract ArrayList<Double> extractFeatures(ArrayList<Double> S);

    public abstract int getFeatureVectorSize(int stateSize);

    public ArrayList<Double> extractFeatures(ArrayList<Double> S, ArrayList<Double> A) {
        ArrayList<Double> newS = new ArrayList<Double>();
        newS.addAll(S);
        newS.addAll(A);

        return extractFeatures(newS);
    }

    public void setMaxStateValues(ArrayList<Double> maxStateValues) {
        this.maxStateValues = maxStateValues;
    }

    protected ArrayList<Double> normalizeValues(ArrayList<Double> S) {
        for (int i = 0; i < Math.min(S.size(), maxStateValues.size()); i++) {
            S.set(i, S.get(i) / maxStateValues.get(i));
        }

        return S;
    }

    // Returns the segments of the jacobian that are obtained through the differentiation with relation
    // to action. Each action has a row, each dimension has a column
    public abstract ArrayList<ArrayList<Double>> getActionJacobian(ArrayList<Double> S, ArrayList<Double> A);
}
