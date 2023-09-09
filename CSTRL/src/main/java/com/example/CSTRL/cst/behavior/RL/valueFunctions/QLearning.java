package com.example.CSTRL.cst.behavior.RL.valueFunctions;

import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
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
        if (!qTable.containsKey(identifier)) {
            qTable.put(identifier, Math.random());
        }
        return qTable.get(identifier);
    }

    @Override
    public Double getValue(ArrayList<Double> state, ArrayList<Double> action) {
        return getQValue(tabularFeatureExtractor.getIdentifier(state, action));
    }

    @Override
    public void update(ArrayList<Double> pastState, ArrayList<Double> pastAction, ArrayList<Double> state, Double reward, Policy policy) {
        String id = tabularFeatureExtractor.getIdentifier(pastState, pastAction);

        // Q(St, At) = Q(St, At) + alpha * (Rt+1 + gamma * Q(St+1, A') - Q(St, At))
        double qValue = getQValue(id);
        double bootstrapValue = reward + discountRate * getValue(state, policy.getPolicyAction(state, this));
        double newQValue = qValue + explorationRate.getRate() * (bootstrapValue - qValue);

        qTable.put(id, newQValue);
        System.out.println(id + " : " + getQValue(id).toString());
    }

    @Override
    public ArrayList<Double> getActionGradient(ArrayList<Double> S, ArrayList<Double> A) {
        return null;
    }
}
