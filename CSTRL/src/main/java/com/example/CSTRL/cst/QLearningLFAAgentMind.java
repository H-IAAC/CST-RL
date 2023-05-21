package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.RLRates.LinearDecreaseRLRate;
import com.example.CSTRL.cst.behavior.RL.actionManagers.ActionManager;
import com.example.CSTRL.cst.behavior.RL.actionManagers.DiscreteActionManager;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.EpsilonGreedyActionSelector;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.DirectFeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.LCFeatureExtractor;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.LFA;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;
import com.example.CSTRL.cst.behavior.StateActionValueFunctionRLCodelet;

import java.util.ArrayList;
import java.util.Arrays;

public class QLearningLFAAgentMind extends AgentMind {

    final double initialEpsilon = 0.2;
    final int episodesToZeroEpsilon = 100;
    final double initialAlpha = 0.001;
    final int episodesToZeroAlpha = 100;
    final double discountRate = 0.9;

    public QLearningLFAAgentMind() {
        super();
    }

    @Override
    protected Codelet getRLCodelet(MemoryObject perceptMO) {
        ArrayList<ArrayList<Double>> actions = new ArrayList<ArrayList<Double>>() {
            {
                add(new ArrayList<Double>(Arrays.asList(1.0, 0.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 1.0, 0.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 1.0, 0.0)));
                add(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 1.0)));
            }
        };
        ActionManager actionManager = new DiscreteActionManager(actions);

        ActionSelector actionSelector = new EpsilonGreedyActionSelector(new LinearDecreaseRLRate(initialEpsilon, episodesToZeroEpsilon));

        StateActionValueFunction stateActionValueFunction = new LFA(new LinearDecreaseRLRate(initialAlpha, episodesToZeroAlpha), discountRate, new DirectFeatureExtractor(), 15);

        return new StateActionValueFunctionRLCodelet(perceptMO, actionManager, actionSelector, stateActionValueFunction);
    }
}
