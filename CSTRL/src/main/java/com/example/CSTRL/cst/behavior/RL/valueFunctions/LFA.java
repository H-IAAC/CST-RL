package com.example.CSTRL.cst.behavior.RL.valueFunctions;

import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.FeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.util.RLMath;

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
            w.add(r.nextDouble() / 10);
        }
    }

    @Override
    public Double getValue(ArrayList<Double> state, ArrayList<Double> action) {
        return RLMath.dotProduct(featureExtractor.extractFeatures(state, action), w);
    }

    @Override
    public void update(ArrayList<Double> pastState, ArrayList<Double> pastAction, ArrayList<Double> state, Double reward, Policy policy) {
        ArrayList<Double> x = featureExtractor.extractFeatures(pastState, pastAction);
        ArrayList<Double> nx = featureExtractor.extractFeatures(state, policy.getPolicyAction(state, this));

        String snx = RLMath.dotProduct(nx, w).toString();
        String sx = RLMath.dotProduct(x, w).toString();

        double delta = explorationRate.getRate() * (reward + discountRate * RLMath.dotProduct(nx, w) - RLMath.dotProduct(x, w));

        for (int i = 0; i < w.size(); i++) {
            w.set(i, w.get(i) + delta * x.get(i));
        }
    }

    @Override
    public ArrayList<Double> getActionGradient(ArrayList<Double> S, ArrayList<Double> A) {
        ArrayList<ArrayList<Double>> featureJacobian = featureExtractor.getActionJacobian(S, A);

        ArrayList<Double> gradient = new ArrayList<Double>();

        for (ArrayList<Double> featureGradient : featureJacobian) {
            gradient.add(RLMath.dotProduct(featureGradient, w));
        }

        return gradient;
    }
}
