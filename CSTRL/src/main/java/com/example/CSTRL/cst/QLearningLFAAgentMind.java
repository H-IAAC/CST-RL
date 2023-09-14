package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.RLRates.LinearDecreaseRLRate;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.policies.DiscretePolicy;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.EpsilonGreedyActionSelector;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.*;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.LFA;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;
import com.example.CSTRL.cst.behavior.StateActionValueFunctionRLCodelet;

import java.util.ArrayList;
import java.util.Arrays;

public class QLearningLFAAgentMind extends AgentMind {

    final double initialEpsilon = 0.9;
    final int episodesToZeroEpsilon = 4000;
    final double initialAlpha = 0.001;
    final int episodesToZeroAlpha = 4000;
    final double discountRate = 0.9;

    public QLearningLFAAgentMind() {
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

        FeatureExtractor featureExtractor = new LCFeatureExtractor(1);
        featureExtractor.setMaxStateValues(new ArrayList<Double>(Arrays.asList(1024.0, 576.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 1.0, 1.0, 1.0, 1.0)));

        StateActionValueFunction stateActionValueFunction = new LFA(new LinearDecreaseRLRate(initialAlpha, episodesToZeroAlpha), discountRate, featureExtractor, 14);

        return new StateActionValueFunctionRLCodelet(perceptMO, policy, actionSelector, stateActionValueFunction);
    }
}
