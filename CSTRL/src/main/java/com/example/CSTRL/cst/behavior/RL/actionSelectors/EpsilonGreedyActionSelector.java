package com.example.CSTRL.cst.behavior.RL.actionSelectors;

import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;
import java.util.Random;

public class EpsilonGreedyActionSelector extends ActionSelector {
    RLRate epsilon;

    public EpsilonGreedyActionSelector(RLRate epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public ArrayList<Double> selectAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction, Policy policy) {
        Random r = new Random();
        if (r.nextDouble() < epsilon.getRate()) {
            return policy.getRandomAction();
        }
        return selectOptimalAction(state, stateActionValueFunction, policy);
    }

    @Override
    public void endEpisode() {
        epsilon.endEpisode();
    }
}
