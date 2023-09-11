package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.RLRates.LinearDecreaseRLRate;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.policies.DiscretePolicy;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.EpsilonGreedyActionSelector;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.FroggerFeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.QLearning;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;
import com.example.CSTRL.cst.behavior.StateActionValueFunctionRLCodelet;

import java.util.ArrayList;
import java.util.Arrays;

public class QLearningAgentMind extends AgentMind {
    final double initialEpsilon = 0.9;
    final int episodesToZeroEpsilon = 15000;
    final double initialAlpha = 0.1;
    final int episodesToZeroAlpha = 15000;
    final double discountRate = 0.9;

    public QLearningAgentMind() {
        super();
    }

    @Override
    protected Codelet getRLCodelet(MemoryObject perceptMO) {
        ArrayList<ArrayList<Double>> actions = new ArrayList<ArrayList<Double>>() {
            {
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(1.0, 0.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 1.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 1.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 1.0)));
            }
        };
        Policy policy = new DiscretePolicy(actions);

        ActionSelector actionSelector = new EpsilonGreedyActionSelector(new LinearDecreaseRLRate(initialEpsilon, episodesToZeroEpsilon));

        StateActionValueFunction stateActionValueFunction = new QLearning(new LinearDecreaseRLRate(initialAlpha, episodesToZeroAlpha), discountRate, new FroggerFeatureExtractor(8, 200, 15, 64));

        return new StateActionValueFunctionRLCodelet(perceptMO, policy, actionSelector, stateActionValueFunction);
    }
}
