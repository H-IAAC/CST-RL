package com.example.CSTRL.cst.behavior.RL.learners;

import com.example.CSTRL.cst.behavior.RL.RLRates.LinearDecreaseRLRate;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.FeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.util.RLMath;
import com.example.CSTRL.util.RLPercept;

import java.util.ArrayList;
import java.util.Random;

public class LFAQLearning extends RLLearner {
    private final FeatureExtractor featureExtractor;
    private final ArrayList<Double> w;
    private final LinearDecreaseRLRate alpha;
    private final Double gamma;
    private final LinearDecreaseRLRate epsilon;


    public LFAQLearning(Double initialAlpha, Double initialEpsilon, int episodesToConverge, Double gamma, FeatureExtractor featureExtractor) {
        this.alpha = new LinearDecreaseRLRate(initialAlpha, episodesToConverge);
        this.epsilon = new LinearDecreaseRLRate(initialEpsilon, episodesToConverge);
        this.gamma = gamma;
        this.featureExtractor = featureExtractor;

        w = new ArrayList<Double>();
    }

    @Override
    public void rlStep(ArrayList<RLPercept> trial) {
        RLPercept pastPercept = trial.get(trial.size() - 2);
        RLPercept currentPercept = trial.get(trial.size() - 1);

        ArrayList<Double> x = featureExtractor.extractFeatures(pastPercept.getState(), pastPercept.getAction());
        addUpToW(x.size());
        ArrayList<Double> nx = featureExtractor.extractFeatures(currentPercept.getState(), selectBestAction(currentPercept.getState()));

        double delta = alpha.getRate() * (currentPercept.getReward() + gamma * RLMath.dotProduct(nx, w) - RLMath.dotProduct(x, w));

        for (int i = 0; i < x.size(); i++) {
            w.set(i, w.get(i) + delta * x.get(i));
        }
    }

    @Override
    public ArrayList<Double> selectAction(ArrayList<Double> s) {
        Random r = new Random();
        if (r.nextDouble() < epsilon.getRate()) {
            return actionSpace.getRandomAction();
        }
        return selectBestAction(s);
    }

    public ArrayList<Double> selectBestAction(ArrayList<Double> s) {
        ArrayList<Double> bestAction = new ArrayList<Double>();
        Double bestValue = Double.NEGATIVE_INFINITY;

        for (ArrayList<Double> action : actionSpace.getDomain()) {
            Double value = getValue(s, action);
            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        }

        return bestAction;
    }

    @Override
    public void endEpisode() {
        alpha.endEpisode();
        epsilon.endEpisode();
    }

    private void addUpToW(int length) {
        if (w.size() < length) {
            Random r = new Random();

            while (w.size() < length) {
                w.add(r.nextDouble());
            }
        }
    }

    public Double getValue(ArrayList<Double> state, ArrayList<Double> action) {
        return RLMath.dotProduct(featureExtractor.extractFeatures(state, action), w);
    }

    @Override
    public String toString() {
        return "LFAQLearning(alpha=[" + alpha.toString() + "],epsilon=[" + epsilon.toString() + "],gamma=[" + gamma.toString() + "],feature_extractor=[" + featureExtractor.toString() + "])";
    }
}
