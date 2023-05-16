package com.example.CSTRL.cst.behavior.RL.valueFunctions;

import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.actionManagers.ActionManager;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.TabularFeatureExtractor;

import java.util.ArrayList;
import java.util.Hashtable;

public class QLearning extends StateActionValueFunction {
    private final Hashtable<String, Double> qTable;
    private final TabularFeatureExtractor tabularFeatureExtractor;

    public QLearning(RLRate explorationRate, Double discountRate, TabularFeatureExtractor tabularFeatureExtractor) {
        super(explorationRate, discountRate);

        this.tabularFeatureExtractor = tabularFeatureExtractor;

        qTable = new Hashtable<>();
    }

    private Double getQValue(String identifier) {
        if (qTable.containsKey(identifier)) {
            return qTable.get(identifier);
        }

        return 0.0;
    }

    @Override
    public Double getValue(ArrayList<Double> state, ArrayList<Double> action) {
        return getQValue(tabularFeatureExtractor.getIdentifier(state, action));
    }

    @Override
    public void update(ArrayList<Double> pastState, ArrayList<Double> pastAction, ArrayList<Double> state, Double reward, ActionManager actionManager) {
        String id = tabularFeatureExtractor.getIdentifier(pastState, pastAction);
        double qValue = getQValue(id);

        // Q(St, At) = Q(St, At) + alpha * (Rt+1 + gamma * Q(St+1, A') - Q(St, At))
        qTable.put(id, qValue + explorationRate.getRate() * (reward + discountRate * getValue(state, actionManager.getBestAction(state, this)) - qValue));
    }
}
