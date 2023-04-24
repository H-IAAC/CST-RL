package com.example.CSTRL.cst.behavior.RL.actionSelectors;

import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.actionManagers.ActionManager;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;
import java.util.Random;

public class EpsilonGreedyActionSelector extends ActionSelector {
    RLRate epsilon;

    public EpsilonGreedyActionSelector(RLRate epsilon) {
        this.epsilon = epsilon;
    }

    @Override
    public ArrayList<Double> selectAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction, ActionManager actionManager) {
        Random r = new Random();
        if (r.nextDouble() < epsilon.getRate()) {
            return actionManager.getRandomAction();
        }
        return selectOptimalAction(state, stateActionValueFunction, actionManager);
    }

    @Override
    public void endEpisode() {
        epsilon.endEpisode();
    }
}
