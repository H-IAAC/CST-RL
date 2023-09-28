package com.example.CSTRL.cst.behavior.RL.learners;

import com.example.CSTRL.cst.behavior.RL.RLRates.LinearDecreaseRLRate;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.TabularFeatureExtractor;
import com.example.CSTRL.util.RLPercept;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Random;

public class QLearning extends RLLearner {

    private final Hashtable<String, Double> qTable;
    private final TabularFeatureExtractor tabularFeatureExtractor;
    private final LinearDecreaseRLRate alpha;
    private final Double gamma;
    private final LinearDecreaseRLRate epsilon;

    public QLearning(Double initialAlpha, Double initialEpsilon, int episodesToConverge, Double gamma, TabularFeatureExtractor tabularFeatureExtractor) {
        this.alpha = new LinearDecreaseRLRate(initialAlpha, episodesToConverge);
        this.epsilon = new LinearDecreaseRLRate(initialEpsilon, episodesToConverge);
        this.gamma = gamma;
        this.tabularFeatureExtractor = tabularFeatureExtractor;

        qTable = new Hashtable<>();
    }

    @Override
    public void rlStep(ArrayList<RLPercept> trial) {
        RLPercept pastPercept = trial.get(trial.size() - 2);
        RLPercept currentPercept = trial.get(trial.size() - 1);

        String id = tabularFeatureExtractor.getIdentifier(pastPercept.getState(), pastPercept.getAction());

        // Q(St, At) = Q(St, At) + alpha * (Rt+1 + gamma * Q(St+1, A') - Q(St, At))
        double qValue = getQValue(id);
        double bootstrapValue = currentPercept.getReward() + gamma * getValue(currentPercept.getState(), selectBestAction(currentPercept.getState()));
        double newQValue = qValue + alpha.getRate() * (bootstrapValue - qValue);

        qTable.put(id, newQValue);
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

    private Double getQValue(String identifier) {
        if (!qTable.containsKey(identifier)) {
            qTable.put(identifier, Math.random());
        }
        return qTable.get(identifier);
    }

    public Double getValue(ArrayList<Double> state, ArrayList<Double> action) {
        return getQValue(tabularFeatureExtractor.getIdentifier(state, action));
    }

    @Override
    public String toString() {
        return "QLearning(alpha=[" + alpha.toString() + "],epsilon=[" + epsilon.toString() + "],gamma=[" + gamma.toString() + "],feature_extractor=[" + tabularFeatureExtractor.toString() + "])";
    }
}
