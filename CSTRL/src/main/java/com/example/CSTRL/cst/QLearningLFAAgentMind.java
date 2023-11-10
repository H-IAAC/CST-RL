package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.EpisodicRLCodelet;
import com.example.CSTRL.cst.behavior.RL.actionSpaces.ActionSpace;
import com.example.CSTRL.cst.behavior.RL.actionSpaces.DiscreteActionSpace;
import com.example.CSTRL.cst.behavior.RL.learners.LFAQLearning;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.*;
import com.example.CSTRL.cst.behavior.RLCodelet;

import java.util.ArrayList;
import java.util.Arrays;

public class QLearningLFAAgentMind extends AgentMind {

    final double initialAlpha = 0.001;
    final double initialEpsilon = 0.9;
    final int episodesToConverge = 4000;
    final double gamma = 0.9;

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
        ActionSpace actionSpace = new DiscreteActionSpace(actions);
        FeatureExtractor featureExtractor = new LCFeatureExtractor(1);
        featureExtractor.setMaxStateValues(new ArrayList<Double>(Arrays.asList(1024.0, 576.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 256.0, 1.0, 1.0, 1.0, 1.0)));
        LFAQLearning lfaQLearning = new LFAQLearning(initialAlpha, initialEpsilon, episodesToConverge, gamma, featureExtractor);

        return new EpisodicRLCodelet(lfaQLearning, actionSpace, perceptMO);
    }
}
