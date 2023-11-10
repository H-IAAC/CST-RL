package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.EpisodicRLCodelet;
import com.example.CSTRL.cst.behavior.RL.actionSpaces.ActionSpace;
import com.example.CSTRL.cst.behavior.RL.actionSpaces.DiscreteActionSpace;
import com.example.CSTRL.cst.behavior.RL.learners.QLearning;
import com.example.CSTRL.cst.behavior.RL.featureExtractors.FroggerFeatureExtractor;

import java.util.ArrayList;
import java.util.Arrays;

public class QLearningAgentMind extends AgentMind {
    final double initialAlpha = 0.1;
    final double initialEpsilon = 0.9;
    final int episodesToConverge = 20000;

    final double gamma = 0.9;

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
        ActionSpace actionSpace = new DiscreteActionSpace(actions);
        QLearning qLearning = new QLearning(initialAlpha, initialEpsilon, episodesToConverge, gamma,
                new FroggerFeatureExtractor(8, 200, 15, 64));
        return new EpisodicRLCodelet(qLearning, actionSpace, perceptMO);
    }
}
