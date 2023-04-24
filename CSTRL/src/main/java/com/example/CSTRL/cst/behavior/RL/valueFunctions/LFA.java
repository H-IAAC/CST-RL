package com.example.CSTRL.cst.behavior.RL.valueFunctions;

import com.example.CSTRL.cst.behavior.RL.actionManagers.ActionManager;
import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.FeatureExtractor;

import java.util.ArrayList;
import java.util.Random;

public class LFA extends StateActionValueFunction {
    FeatureExtractor featureExtractor;
    ArrayList<Double> w;

    public LFA(RLRate explorationRate, Double discountRate, FeatureExtractor featureExtractor, int stateActionSize) {
        super(explorationRate, discountRate);

        this.featureExtractor = featureExtractor;

        w = new ArrayList<Double>();
        Random r = new Random();
        for (int i = 0; i < featureExtractor.getFeatureVectorSize(stateActionSize); i++) {
            w.add(r.nextDouble());
        }
    }

    private static Double dotProduct(ArrayList<Double> a1, ArrayList<Double> a2) {
        if (a1.size() != a2.size()) {
            throw new RuntimeException("Arrays must be of same size");
        }

        double sum = 0.0;
        for (int i = 0; i < a1.size(); i++) {
            sum += a1.get(i) * a2.get(i);
        }

        return sum;
    }

    @Override
    public Double getValue(ArrayList<Double> state, ArrayList<Double> action) {
        return dotProduct(featureExtractor.extractFeatures(state, action), w);
    }

    @Override
    public void update(ArrayList<Double> past_state, ArrayList<Double> past_action, ArrayList<Double> state, Double reward, ActionManager actionManager) {
        ArrayList<Double> x = featureExtractor.extractFeatures(past_state, past_action);
        ArrayList<Double> nx = featureExtractor.extractFeatures(state, actionManager.getBestAction(state, this));

        double delta = explorationRate.getRate() * (reward + discountRate * dotProduct(nx, w) - dotProduct(x, w));
        for (int i = 0; i < w.size(); i++) {
            w.set(i, w.get(i) + delta * x.get(i));
        }
    }
}
