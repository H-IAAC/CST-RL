package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.RLRates.LinearDecreaseRLRate;
import com.example.CSTRL.cst.behavior.RL.policies.ContinuousPolicy;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.EpsilonGreedyActionSelector;
import com.example.CSTRL.cst.behavior.RL.actionTranslators.SimplifiedActionTranslator;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.SimplifiedLFAFeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.policies.DiscretePolicy;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.LFA;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;
import com.example.CSTRL.cst.behavior.RLCodelet;
import com.example.CSTRL.cst.behavior.StateActionValueFunctionRLCodelet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SimplifiedLFAAgentMind extends AgentMind {
    final double initialEpsilon = 0.9;
    final int episodesToZeroEpsilon = 500;
    final double initialAlpha = 0.001;
    final int episodesToZeroAlpha = 500;
    final double discountRate = 0.9;

    @Override
    protected Codelet getRLCodelet(MemoryObject perceptMO) {
        ArrayList<ArrayList<Double>> actions = new ArrayList<ArrayList<Double>>() {
            {
                add(new ArrayList<Double>(List.of(1.0)));
                add(new ArrayList<Double>(List.of(0.75)));
                add(new ArrayList<Double>(List.of(0.5)));
                add(new ArrayList<Double>(List.of(0.25)));
                add(new ArrayList<Double>(List.of(0.0)));
                add(new ArrayList<Double>(List.of(-0.25)));
                add(new ArrayList<Double>(List.of(-0.5)));
                add(new ArrayList<Double>(List.of(-0.75)));
                add(new ArrayList<Double>(List.of(-1.0)));
            }
        };
        Policy policy = new DiscretePolicy(actions);

        ActionSelector actionSelector = new EpsilonGreedyActionSelector(new LinearDecreaseRLRate(initialEpsilon, episodesToZeroEpsilon));

        SimplifiedLFAFeatureExtractor featureExtractor = new SimplifiedLFAFeatureExtractor(8);
        featureExtractor.setMaxStateValues(new ArrayList<>(Arrays.asList(1024.0, 576.0, 6.2832, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 1.0)));

        StateActionValueFunction stateActionValueFunction = new LFA(new LinearDecreaseRLRate(initialAlpha, episodesToZeroAlpha), discountRate, featureExtractor, 12);

        RLCodelet rlCodelet = new StateActionValueFunctionRLCodelet(perceptMO, policy, actionSelector, stateActionValueFunction);
        rlCodelet.setActionTranslator(new SimplifiedActionTranslator());

        return rlCodelet;
    }
}
